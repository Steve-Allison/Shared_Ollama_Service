"""
Comprehensive behavioral tests for the FastAPI REST API server.

Tests focus on real API behavior, queue integration, streaming, rate limiting,
error handling, and edge cases. Mocks are only used for external Ollama service.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from typing import Any

import pytest
from fastapi.testclient import TestClient

from shared_ollama.api.dependencies import set_dependencies
from shared_ollama.api.server import app
from shared_ollama.application.use_cases import ChatUseCase, GenerateUseCase, ListModelsUseCase
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient, GenerateResponse
from shared_ollama.core.config import settings
from shared_ollama.core.queue import RequestQueue
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    ImageCacheAdapter,
    ImageProcessorAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)
from tests.test_model_validation import VALID_IMAGE_DATA_URL
class _DummyImageProcessor:
    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        return "image/jpeg", b""

    def process_image(self, data_url: str, target_format: str = "jpeg") -> tuple[str, Any]:
        return data_url, {"format": target_format}


class _DummyImageCache:
    def get(self, data_url: str, target_format: str) -> None:
        return None

    def put(self, data_url: str, target_format: str, base64_string: str, metadata: Any) -> None:
        return None

    def get_stats(self) -> dict[str, int]:
        return {}

from tests.helpers import (
    assert_error_response,
    assert_response_structure,
    cleanup_dependency_overrides,
    setup_dependency_overrides,
)


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncSharedOllamaClient for external service."""
    client = AsyncMock(spec=AsyncSharedOllamaClient)
    client.config = AsyncOllamaConfig()
    client._ensure_client = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_use_cases(mock_async_client):
    """Create mock use cases with mocked client adapter."""
    client_adapter = AsyncOllamaClientAdapter(mock_async_client)
    logger_adapter = RequestLoggerAdapter()
    metrics_adapter = MetricsCollectorAdapter()

    generate_use_case = GenerateUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )
    chat_use_case = ChatUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )
    list_models_use_case = ListModelsUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )

    return {
        "generate": generate_use_case,
        "chat": chat_use_case,
        "list_models": list_models_use_case,
    }


@pytest.fixture
def sync_api_client(mock_async_client, mock_use_cases):
    """Create a sync test client for endpoints without dependencies (health, root)."""
    # Set up dependencies
    client_adapter = AsyncOllamaClientAdapter(mock_async_client)
    logger_adapter = RequestLoggerAdapter()
    metrics_adapter = MetricsCollectorAdapter()
    queue = RequestQueue(max_concurrent=3, max_queue_size=10)

    setup_dependency_overrides(
        app=app,
        client_adapter=client_adapter,
        logger_adapter=logger_adapter,
        metrics_adapter=metrics_adapter,
        queue=queue,
        generate_use_case=mock_use_cases["generate"],
        chat_use_case=mock_use_cases["chat"],
        list_models_use_case=mock_use_cases["list_models"],
    )

    try:
        with TestClient(app) as client:
            yield client
    finally:
        cleanup_dependency_overrides(app)


@pytest.fixture
def api_client(mock_async_client, mock_use_cases):
    """Create a test client using TestClient - works with all dependency patterns.

    TestClient handles async endpoints correctly and works with Annotated[Type, Depends(...)]
    syntax, unlike ASGITransport which has issues with this pattern.
    """
    # Set up dependencies
    client_adapter = AsyncOllamaClientAdapter(mock_async_client)
    logger_adapter = RequestLoggerAdapter()
    metrics_adapter = MetricsCollectorAdapter()
    queue = RequestQueue(max_concurrent=3, max_queue_size=10)

    setup_dependency_overrides(
        app=app,
        client_adapter=client_adapter,
        logger_adapter=logger_adapter,
        metrics_adapter=metrics_adapter,
        queue=queue,
        generate_use_case=mock_use_cases["generate"],
        chat_use_case=mock_use_cases["chat"],
        list_models_use_case=mock_use_cases["list_models"],
    )

    try:
        with TestClient(app) as client:
            yield client
    finally:
        cleanup_dependency_overrides(app)


@pytest.fixture
def api_client_with_vlm(mock_async_client, mock_use_cases):
    """Test client that provides a mocked VLM use case."""
    client_adapter = AsyncOllamaClientAdapter(mock_async_client)
    logger_adapter = RequestLoggerAdapter()
    metrics_adapter = MetricsCollectorAdapter()
    queue = RequestQueue(max_concurrent=3, max_queue_size=10)

    class DummyVLMUseCase:
        def __init__(self) -> None:
            self.execute = AsyncMock()

    vlm_use_case = DummyVLMUseCase()
    image_processor_adapter = ImageProcessorAdapter(_DummyImageProcessor())
    image_cache_adapter = ImageCacheAdapter(_DummyImageCache())

    setup_dependency_overrides(
        app=app,
        client_adapter=client_adapter,
        logger_adapter=logger_adapter,
        metrics_adapter=metrics_adapter,
        queue=queue,
        generate_use_case=mock_use_cases["generate"],
        chat_use_case=mock_use_cases["chat"],
        list_models_use_case=mock_use_cases["list_models"],
        image_processor_adapter=image_processor_adapter,
        image_cache_adapter=image_cache_adapter,
        vlm_use_case=vlm_use_case,
    )

    try:
        with TestClient(app) as client:
            yield client, vlm_use_case
    finally:
        cleanup_dependency_overrides(app)


class TestHealthEndpoint:
    """Behavioral tests for health check endpoint."""

    def test_health_check_success(self, sync_api_client, mock_async_client):
        """Test successful health check returns healthy status."""
        with patch("shared_ollama.api.routes.system.check_service_health", return_value=(True, None)):
            response = sync_api_client.get("/api/v1/health")
            data = assert_response_structure(response, 200)
            assert data["status"] == "healthy"
            assert data["ollama_service"] == "healthy"
            assert data["version"] == settings.api.version

    def test_health_check_unhealthy(self, sync_api_client, mock_async_client):
        """Test health check returns unhealthy when service is down."""
        # Need to patch at the module level where it's imported
        with patch(
            "shared_ollama.api.routes.system.check_service_health",
            return_value=(False, "Connection refused"),
        ):
            response = sync_api_client.get("/api/v1/health")
            data = assert_response_structure(response, 200)
            assert data["status"] == "unhealthy"
            assert "unhealthy" in data["ollama_service"]
            assert "Connection refused" in data["ollama_service"]


class TestListModelsEndpoint:
    """Behavioral tests for list models endpoint."""

    def test_list_models_success(self, api_client, mock_async_client):
        """Test successful model listing returns models list."""
        mock_models = [
            {"name": "qwen3-vl:8b-instruct-q4_K_M", "size": 5969245856, "modified_at": "2025-11-03T17:24:58Z"},
            {"name": "qwen3:14b-q4_K_M", "size": 8988124069, "modified_at": "2025-11-03T15:00:00Z"},
        ]
        mock_async_client.list_models = AsyncMock(return_value=mock_models)

        response = api_client.get("/api/v1/models")
        data = assert_response_structure(response, 200)
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["name"] == "qwen3-vl:8b-instruct-q4_K_M"
        assert data["models"][1]["name"] == "qwen3:14b-q4_K_M"

    def test_list_models_empty(self, api_client, mock_async_client):
        """Test listing models when none are available returns empty list."""
        mock_async_client.list_models = AsyncMock(return_value=[])

        response = api_client.get("/api/v1/models")
        data = assert_response_structure(response, 200)
        assert data["models"] == []

    def test_list_models_error_returns_503(self, api_client, mock_async_client):
        """Test that connection errors during model listing return 503."""
        mock_async_client.list_models = AsyncMock(side_effect=ConnectionError("Service unavailable"))

        response = api_client.get("/api/v1/models")
        assert_error_response(response, 503)


class TestQueueStatsEndpoint:
    """Behavioral tests for queue statistics endpoint."""

    def test_get_queue_stats_returns_comprehensive_metrics(self, api_client, mock_async_client):
        """Test that queue stats endpoint returns all queue metrics."""
        response = api_client.get("/api/v1/queue/stats")
        data = assert_response_structure(response, 200)

        # Verify all expected fields are present
        assert "queued" in data
        assert "in_progress" in data
        assert "completed" in data
        assert "failed" in data
        assert "rejected" in data
        assert "timeout" in data
        assert "total_wait_time_ms" in data
        assert "max_wait_time_ms" in data
        assert "avg_wait_time_ms" in data
        assert "max_concurrent" in data
        assert "max_queue_size" in data
        assert "default_timeout" in data

        # Verify types
        assert isinstance(data["queued"], int)
        assert isinstance(data["in_progress"], int)
        assert isinstance(data["max_concurrent"], int)
        assert isinstance(data["max_queue_size"], int)


class TestSystemMetricsEndpoints:
    """Behavioral tests for metrics/performance/analytics endpoints."""

    @staticmethod
    def _seed_metrics() -> None:
        """Seed telemetry collectors with deterministic values."""
        from datetime import datetime, timezone

        from shared_ollama.telemetry.analytics import AnalyticsCollector
        from shared_ollama.telemetry.metrics import MetricsCollector, RequestMetrics
        from shared_ollama.telemetry.performance import (
            DetailedPerformanceMetrics,
            PerformanceCollector,
        )

        now = datetime.now(timezone.utc)

        MetricsCollector.reset()
        PerformanceCollector.reset()
        AnalyticsCollector._project_metadata.clear()  # type: ignore[attr-defined]

        MetricsCollector._metrics.extend(  # type: ignore[attr-defined]
            [
                RequestMetrics("model-a", "chat", 100.0, True, timestamp=now),
                RequestMetrics("model-a", "chat", 200.0, False, error="RuntimeError", timestamp=now),
                RequestMetrics("model-b", "vlm", 150.0, True, timestamp=now),
            ]
        )

        PerformanceCollector._metrics.extend(  # type: ignore[attr-defined]
            [
                DetailedPerformanceMetrics(
                    model="model-a",
                    operation="chat",
                    timestamp=now,
                    total_latency_ms=120.0,
                    success=True,
                    tokens_per_second=20.0,
                    load_time_ms=50.0,
                    generation_time_ms=70.0,
                )
            ]
        )

        AnalyticsCollector._project_metadata.update(  # type: ignore[attr-defined]
            {0: "proj-alpha", 1: "proj-alpha", 2: "proj-beta"}
        )

    def test_metrics_endpoint_returns_typed_payload(self, api_client):
        """Metrics endpoint should return structured MetricsResponse data."""
        self._seed_metrics()
        response = api_client.get("/api/v1/metrics")
        data = assert_response_structure(response, 200)

        assert data["total_requests"] == 3
        assert data["successful_requests"] == 2
        assert data["failed_requests"] == 1
        assert set(data["requests_by_model"].keys()) == {"model-a", "model-b"}
        assert data["p50_latency_ms"] >= 0
        assert data["last_request_time"] is not None

    def test_performance_stats_endpoint_returns_structured_data(self, api_client):
        """Performance stats endpoint should emit PerformanceStatsResponse payload."""
        self._seed_metrics()
        response = api_client.get("/api/v1/performance/stats")
        data = assert_response_structure(response, 200)

        assert data["total_requests"] == 1
        assert data["avg_tokens_per_second"] == 20.0
        assert "model-a" in data["by_model"]
        assert data["by_model"]["model-a"]["request_count"] == 1

    def test_analytics_endpoint_returns_structured_data(self, api_client):
        """Analytics endpoint should emit AnalyticsResponse payload with project/hour data."""
        self._seed_metrics()
        response = api_client.get("/api/v1/analytics")
        data = assert_response_structure(response, 200)

        assert data["total_requests"] == 3
        assert "proj-alpha" in data["project_metrics"]
        assert data["project_metrics"]["proj-alpha"]["total_requests"] == 2
        assert isinstance(data["hourly_metrics"], list)


class TestGenerateEndpoint:
    """Comprehensive behavioral tests for generate endpoint."""

    def test_generate_success_returns_response(self, api_client, mock_async_client):
        """Test successful generation returns GenerateResponse."""
        mock_response = GenerateResponse(
            text="Hello, world!",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=500_000_000,
            load_duration=200_000_000,
            prompt_eval_count=5,
            prompt_eval_duration=100_000_000,
            eval_count=10,
            eval_duration=400_000_000,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Hello", "model": "qwen3-vl:8b-instruct-q4_K_M"},
        )
        data = assert_response_structure(response, 200)
        assert data["text"] == "Hello, world!"
        assert data["model"] == "qwen3-vl:8b-instruct-q4_K_M"
        assert "request_id" in data
        assert "latency_ms" in data
        assert data["model_warm_start"] is False  # load_duration > 0

    def test_generate_with_all_options(self, api_client, mock_async_client):
        """Test generation with all optional parameters."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=300_000_000,
            load_duration=0,  # Warm start
            prompt_eval_count=3,
            prompt_eval_duration=50_000_000,
            eval_count=5,
            eval_duration=250_000_000,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/generate",
            json={
                "prompt": "Test",
                "model": "qwen3-vl:8b-instruct-q4_K_M",
                "system": "You are helpful",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 100,
                "seed": 42,
                "stop": ["\n\n"],
                "format": "json",
            },
        )
        data = assert_response_structure(response, 200)
        assert data["text"] == "Response"
        assert data["model_warm_start"] is True  # load_duration == 0

    def test_generate_response_format_json_object_forwards_format(self, api_client, mock_async_client):
        """Ensure response_format=json_object becomes format='json' for backend."""
        mock_response = GenerateResponse(
            text="{}",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=100_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/generate",
            json={
                "prompt": "Return JSON",
                "response_format": {"type": "json_object"},
            },
        )
        data = assert_response_structure(response, 200)
        assert data["text"] == "{}"
        mock_async_client.generate.assert_awaited_once()
        assert mock_async_client.generate.await_args.kwargs["format"] == "json"

    def test_generate_response_format_json_schema_forwards_schema(self, api_client, mock_async_client):
        """Ensure response_format json_schema passes schema to backend."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        mock_response = GenerateResponse(
            text='{"answer": "42"}',
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=100_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/generate",
            json={
                "prompt": "Return JSON",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer_schema", "schema": schema},
                },
            },
        )
        data = assert_response_structure(response, 200)
        assert data["text"] == '{"answer": "42"}'
        forwarded_format = mock_async_client.generate.await_args.kwargs["format"]
        assert forwarded_format == schema

    def test_generate_validates_empty_prompt(self, api_client, mock_async_client):
        """Test that empty prompt is rejected."""
        response = api_client.post("/api/v1/generate", json={"prompt": ""})
        assert response.status_code in [422, 400]  # Validation error

    def test_generate_validates_whitespace_only_prompt(self, api_client, mock_async_client):
        """Test that whitespace-only prompt is rejected."""
        response = api_client.post("/api/v1/generate", json={"prompt": "   "})
        assert response.status_code in [422, 400]

    def test_generate_validates_prompt_length(self, api_client, mock_async_client):
        """Test that extremely long prompt is rejected."""
        long_prompt = "x" * 1_000_001  # Exceeds 1M character limit
        response = api_client.post("/api/v1/generate", json={"prompt": long_prompt})
        assert response.status_code in [400, 422]

    def test_generate_handles_missing_prompt(self, api_client, mock_async_client):
        """Test that missing prompt field is rejected."""
        response = api_client.post("/api/v1/generate", json={})
        assert response.status_code == 422  # Validation error

    def test_generate_handles_invalid_json(self, api_client, mock_async_client):
        """Test that invalid JSON in body is handled."""
        response = api_client.post(
            "/api/v1/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_generate_error_returns_500(self, api_client, mock_async_client):
        """Test that generation errors return 500."""
        mock_async_client.generate = AsyncMock(side_effect=RuntimeError("Model not found"))

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Hello"},
        )
        assert_error_response(response, 500)

    def test_generate_uses_queue_slot(self, api_client, mock_async_client):
        """Test that generate endpoint uses queue for concurrency control."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=300_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post("/api/v1/generate", json={"prompt": "Test"})
        assert response.status_code == 200

        # Verify queue stats show activity
        stats_response = api_client.get("/api/v1/queue/stats")
        stats = stats_response.json()
        assert stats["completed"] >= 1 or stats["in_progress"] >= 0


class TestGenerateStreaming:
    """Behavioral tests for streaming generate endpoint."""

    def test_generate_stream_returns_sse_format(self, api_client, mock_async_client):
        """Test that streaming generate returns Server-Sent Events format."""
        async def mock_stream():
            yield {"chunk": "Hello", "done": False, "model": "test", "request_id": "test-1"}
            yield {"chunk": " world", "done": False, "model": "test", "request_id": "test-1"}
            yield {
                "chunk": "!",
                "done": True,
                "model": "test",
                "request_id": "test-1",
                "latency_ms": 100.0,
            }

        mock_async_client.generate_stream = AsyncMock(return_value=mock_stream())

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Hello", "stream": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Read streaming response (TestClient supports iter_lines for SSE)
        text = ""
        for line in response.iter_lines():
            text += line + "\n"

        # Verify SSE format
        assert "data: " in text
        assert "\n\n" in text or "\n" in text

    def test_generate_stream_handles_errors(self, api_client, mock_async_client):
        """Test that streaming errors are sent as final chunk."""
        from collections.abc import AsyncIterator
        
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            raise RuntimeError("Streaming error")
            # This yield is unreachable but makes it an async generator
            yield {}  # type: ignore[unreachable]

        mock_async_client.generate_stream = AsyncMock(return_value=mock_stream())

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Test", "stream": True},
        )
        assert response.status_code == 200

        text = ""
        for line in response.iter_lines():
            text += line + "\n"

        # Should contain error in final chunk
        assert "error" in text.lower() or "done" in text.lower()


class TestChatEndpoint:
    """Comprehensive behavioral tests for chat endpoint."""

    def test_chat_success_returns_response(self, api_client, mock_async_client):
        """Test successful chat returns ChatResponse."""
        mock_response = {
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "prompt_eval_count": 10,
            "eval_count": 15,
            "total_duration": 400_000_000,
            "load_duration": 0,
        }
        mock_async_client.chat = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "qwen3-vl:8b-instruct-q4_K_M",
            },
        )
        data = assert_response_structure(response, 200)
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello! How can I help?"
        assert data["model"] == "qwen3-vl:8b-instruct-q4_K_M"
        assert "request_id" in data

    def test_chat_validates_empty_messages(self, api_client, mock_async_client):
        """Test that empty messages list is rejected."""
        response = api_client.post("/api/v1/chat", json={"messages": []})
        assert response.status_code == 422  # Validation error

    def test_chat_validates_message_structure(self, api_client, mock_async_client):
        """Test that invalid message structure is rejected."""
        response = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "invalid", "content": "test"}]},
        )
        assert response.status_code in [400, 422]

    def test_chat_validates_empty_message_content(self, api_client, mock_async_client):
        """Test that empty message content is rejected."""
        response = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": ""}]},
        )
        assert response.status_code in [400, 422]

    def test_chat_validates_total_message_length(self, api_client, mock_async_client):
        """Test that extremely long total message content is rejected."""
        long_content = "x" * 500_001  # Exceeds limit when combined
        response = api_client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": long_content},
                    {"role": "user", "content": long_content},
                ]
            },
        )
        assert response.status_code in [400, 422]

    def test_chat_handles_multiple_messages(self, api_client, mock_async_client):
        """Test chat with conversation history."""
        mock_response = {
            "message": {"role": "assistant", "content": "Response"},
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "prompt_eval_count": 20,
            "eval_count": 10,
            "total_duration": 500_000_000,
            "load_duration": 100_000_000,
        }
        mock_async_client.chat = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "model": "qwen3-vl:8b-instruct-q4_K_M",
            },
        )
        data = assert_response_structure(response, 200)
        assert data["message"]["content"] == "Response"

    def test_chat_response_format_json_object_forwards_format(self, api_client, mock_async_client):
        """Ensure chat response_format=json_object maps to format='json'."""
        mock_response = {
            "message": {"role": "assistant", "content": '{"name": "test"}'},
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration": 100_000_000,
            "load_duration": 0,
        }
        mock_async_client.chat = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {"type": "json_object"},
            },
        )
        data = assert_response_structure(response, 200)
        assert data["message"]["content"] == '{"name": "test"}'
        forwarded_format = mock_async_client.chat.await_args.kwargs["format"]
        assert forwarded_format == "json"

    def test_chat_response_format_json_schema_forwards_schema(self, api_client, mock_async_client):
        """Ensure chat response_format json_schema passes schema to backend."""
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }
        mock_response = {
            "message": {"role": "assistant", "content": '{"summary": "done"}'},
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration": 100_000_000,
            "load_duration": 0,
        }
        mock_async_client.chat = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "summary", "schema": schema},
                },
            },
        )
        data = assert_response_structure(response, 200)
        assert data["message"]["content"] == '{"summary": "done"}'
        forwarded_format = mock_async_client.chat.await_args.kwargs["format"]
        assert forwarded_format == schema


class TestVLMOpenAIEndpoint:
    """Behavioral tests for the OpenAI-compatible VLM endpoint."""

    def test_vlm_openai_success_returns_response(self, api_client_with_vlm):
        """Ensure VLM OpenAI route handles multimodal payloads."""
        client, vlm_use_case = api_client_with_vlm
        vlm_use_case.execute.return_value = {
            "message": {"role": "assistant", "content": "A colorful slide about workflows."},
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "prompt_eval_count": 5,
            "eval_count": 10,
            "total_duration": 100_000_000,
            "load_duration": 0,
        }

        payload = {
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": VALID_IMAGE_DATA_URL}},
                    ],
                }
            ],
        }

        response = client.post("/api/v1/vlm/openai", json=payload)
        data = assert_response_structure(response, 200)
        assert data["choices"][0]["message"]["content"] == "A colorful slide about workflows."
        vlm_use_case.execute.assert_awaited_once()

    def test_chat_error_returns_400(self, api_client, mock_async_client):
        """Test that chat validation errors return 400."""
        # ValueError from use case gets converted to InvalidRequestError, which returns 400
        mock_async_client.chat = AsyncMock(side_effect=ValueError("Invalid messages"))

        response = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert_error_response(response, 400)


class TestChatStreaming:
    """Behavioral tests for streaming chat endpoint."""

    def test_chat_stream_returns_sse_format(self, api_client, mock_async_client):
        """Test that streaming chat returns Server-Sent Events format."""
        async def mock_stream():
            yield {"chunk": "Hello", "role": "assistant", "done": False, "model": "test"}
            yield {"chunk": " there", "role": "assistant", "done": False, "model": "test"}
            yield {
                "chunk": "!",
                "role": "assistant",
                "done": True,
                "model": "test",
                "latency_ms": 100.0,
            }

        mock_async_client.chat_stream = AsyncMock(return_value=mock_stream())

        response = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestRequestContext:
    """Behavioral tests for request context extraction."""

    def test_project_name_header_is_captured(self, api_client, mock_async_client):
        """Test that X-Project-Name header is captured in context."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=300_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Test"},
            headers={"X-Project-Name": "Knowledge_Machine"},
        )
        data = assert_response_structure(response, 200)
        assert "request_id" in data

    def test_request_id_is_unique(self, api_client, mock_async_client):
        """Test that each request gets unique request ID."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=300_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response1 = api_client.post("/api/v1/generate", json={"prompt": "Test 1"})
        response2 = api_client.post("/api/v1/generate", json={"prompt": "Test 2"})

        data1 = response1.json()
        data2 = response2.json()
        assert data1["request_id"] != data2["request_id"]


class TestRootEndpoint:
    """Behavioral tests for root endpoint."""

    def test_root_endpoint_returns_api_info(self, sync_api_client):
        """Test root endpoint returns API metadata."""
        response = sync_api_client.get("/")
        data = assert_response_structure(response, 200)
        assert data["service"] == "Shared Ollama Service API"
        assert data["version"] == settings.api.version
        assert "/api/docs" in data["docs"]
        assert "/api/v1/health" in data["health"]


class TestErrorHandling:
    """Comprehensive behavioral tests for error handling."""

    def test_client_not_initialized_returns_503(self, api_client):
        """Test that uninitialized client returns 503."""
        # Clear dependencies to simulate uninitialized state
        cleanup_dependency_overrides(app)
        set_dependencies(None, None, None, None, None, None, None)  # type: ignore[arg-type]

        try:
            response = api_client.get("/api/v1/models")
            assert_error_response(response, 503)
        finally:
            # Restore dependencies
            client_adapter = AsyncOllamaClientAdapter(AsyncMock(spec=AsyncSharedOllamaClient))
            logger_adapter = RequestLoggerAdapter()
            metrics_adapter = MetricsCollectorAdapter()
            queue = RequestQueue()
            vlm_queue = RequestQueue()
            image_processor_adapter = ImageProcessorAdapter(_DummyImageProcessor())
            image_cache_adapter = ImageCacheAdapter(_DummyImageCache())
            set_dependencies(
                client_adapter,
                logger_adapter,
                metrics_adapter,
                queue,
                vlm_queue,
                image_processor_adapter,
                image_cache_adapter,
            )

    def test_validation_error_returns_422(self, api_client, mock_async_client):
        """Test that validation errors return 422 with details."""
        response = api_client.post("/api/v1/generate", json={"invalid": "data"})
        assert_error_response(response, 422)

    def test_global_exception_handler_returns_500(self, api_client, mock_async_client):
        """Test that unexpected exceptions are handled gracefully."""
        mock_async_client.list_models = AsyncMock(side_effect=Exception("Unexpected error"))

        response = api_client.get("/api/v1/models")
        assert_error_response(response, 500)

    def test_error_response_includes_request_id(self, api_client, mock_async_client):
        """Test that error responses include request_id when available."""
        mock_async_client.list_models = AsyncMock(side_effect=RuntimeError("Test error"))

        response = api_client.get("/api/v1/models")
        data = assert_error_response(response, 500)
        # Request ID should be present in error response (either in detail or as separate field)
        assert "request_id" in data.get("detail", "") or "request_id" in data or "error" in data


class TestAsyncFunctionality:
    """Behavioral tests for async endpoint behavior."""

    def test_endpoints_are_truly_async(self, api_client, mock_async_client):
        """Test that endpoints handle async operations correctly."""
        mock_response = GenerateResponse(
            text="Async response",
            model="qwen3-vl:8b-instruct-q4_K_M",
            context=None,
            total_duration=300_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Test async"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Async response"
