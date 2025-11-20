"""Reusable test utilities and helpers for Shared Ollama Service tests.

This module provides common patterns and utilities used across test files,
promoting code reuse and consistency.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient, Response

from shared_ollama.api.dependencies import (
    get_chat_queue,
    get_chat_use_case,
    get_client_adapter,
    get_generate_use_case,
    get_list_models_use_case,
    get_logger_adapter,
    get_metrics_adapter,
    get_vlm_queue,
)
from shared_ollama.core.queue import RequestQueue
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)


def setup_dependency_overrides(
    app: FastAPI,
    client_adapter: AsyncOllamaClientAdapter,
    logger_adapter: RequestLoggerAdapter,
    metrics_adapter: MetricsCollectorAdapter,
    queue: RequestQueue,
    generate_use_case: Any,
    chat_use_case: Any,
    list_models_use_case: Any,
) -> None:
    """Set up FastAPI dependency overrides for testing.

    This function configures all dependency overrides needed for testing,
    including nested dependencies.

    Args:
        app: FastAPI application instance.
        client_adapter: Ollama client adapter.
        logger_adapter: Request logger adapter.
        metrics_adapter: Metrics collector adapter.
        queue: Request queue instance (used for both chat and VLM in tests).
        generate_use_case: Generate use case instance.
        chat_use_case: Chat use case instance.
        list_models_use_case: List models use case instance.
    """
    # Override all dependencies in the chain
    app.dependency_overrides[get_client_adapter] = lambda: client_adapter
    app.dependency_overrides[get_logger_adapter] = lambda: logger_adapter
    app.dependency_overrides[get_metrics_adapter] = lambda: metrics_adapter
    app.dependency_overrides[get_generate_use_case] = lambda: generate_use_case
    app.dependency_overrides[get_chat_use_case] = lambda: chat_use_case
    app.dependency_overrides[get_list_models_use_case] = lambda: list_models_use_case
    # Use same queue for both chat and VLM in tests for simplicity
    app.dependency_overrides[get_chat_queue] = lambda: queue
    app.dependency_overrides[get_vlm_queue] = lambda: queue


def cleanup_dependency_overrides(app: FastAPI) -> None:
    """Clean up FastAPI dependency overrides after testing.

    Args:
        app: FastAPI application instance.
    """
    app.dependency_overrides.clear()


def assert_response_structure(response: Response, expected_status: int = 200) -> dict[str, Any]:
    """Assert response has expected status and return JSON data.

    Args:
        response: HTTP response object.
        expected_status: Expected HTTP status code.

    Returns:
        Parsed JSON response data.

    Raises:
        AssertionError: If status code doesn't match or response isn't JSON.
    """
    assert response.status_code == expected_status, (
        f"Expected status {expected_status}, got {response.status_code}. "
        f"Response: {response.text}"
    )
    assert response.headers.get("content-type", "").startswith("application/json")
    return response.json()


def assert_error_response(
    response: Response,
    expected_status: int,
    error_key: str = "error",
) -> dict[str, Any]:
    """Assert error response has expected structure.

    Args:
        response: HTTP response object.
        expected_status: Expected HTTP status code.
        error_key: Key in response JSON containing error message.

    Returns:
        Parsed JSON response data.

    Raises:
        AssertionError: If response doesn't match expected error format.
    """
    data = assert_response_structure(response, expected_status)
    assert error_key in data or "detail" in data, f"Error response missing '{error_key}' key"
    return data


def create_mock_generate_response(
    text: str = "Hello, world!",
    model: str = "qwen3-vl:8b-instruct-q4_K_M",
    total_duration: int = 500_000_000,
    load_duration: int = 200_000_000,
    prompt_eval_count: int = 5,
    eval_count: int = 10,
) -> dict[str, Any]:
    """Create a mock generate response dictionary.

    Args:
        text: Generated text.
        model: Model name.
        total_duration: Total duration in nanoseconds.
        load_duration: Load duration in nanoseconds.
        prompt_eval_count: Prompt evaluation count.
        eval_count: Generation evaluation count.

    Returns:
        Dictionary matching GenerateResponse structure.
    """
    return {
        "text": text,
        "model": model,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "total_duration": total_duration,
        "load_duration": load_duration,
    }


def create_mock_chat_response(
    content: str = "Hello! How can I help you today?",
    model: str = "qwen3-vl:8b-instruct-q4_K_M",
    total_duration: int = 500_000_000,
    load_duration: int = 200_000_000,
    prompt_eval_count: int = 5,
    eval_count: int = 10,
) -> dict[str, Any]:
    """Create a mock chat response dictionary.

    Args:
        content: Assistant message content.
        model: Model name.
        total_duration: Total duration in nanoseconds.
        load_duration: Load duration in nanoseconds.
        prompt_eval_count: Prompt evaluation count.
        eval_count: Generation evaluation count.

    Returns:
        Dictionary matching ChatResponse structure.
    """
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def create_mock_models_response() -> list[dict[str, Any]]:
    """Create a mock models list response.

    Returns:
        List of model dictionaries.
    """
    return [
        {
            "name": "qwen3-vl:8b-instruct-q4_K_M",
            "size": 5969245856,
            "modified_at": "2025-11-03T17:24:58Z",
        },
        {
            "name": "qwen3:14b-q4_K_M",
            "size": 8988124069,
            "modified_at": "2025-11-03T15:00:00Z",
        },
    ]


async def read_streaming_response(
    client: AsyncClient,
    endpoint: str,
    json_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read and parse a streaming SSE response.

    Args:
        client: AsyncClient instance.
        json_data: Optional JSON data for POST requests.

    Returns:
        List of parsed JSON chunks from the stream.

    Raises:
        AssertionError: If response format is invalid.
    """
    if json_data:
        async with client.stream("POST", endpoint, json=json_data) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_data = line[6:]  # Remove "data: " prefix
                    if chunk_data.strip():
                        import json

                        chunks.append(json.loads(chunk_data))
            return chunks
    else:
        async with client.stream("GET", endpoint) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip():
                        import json

                        chunks.append(json.loads(chunk_data))
            return chunks
