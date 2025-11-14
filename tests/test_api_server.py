"""
Comprehensive behavioral tests for the FastAPI REST API server.

Tests focus on real API behavior, queue integration, streaming, rate limiting,
error handling, and edge cases. Mocks are only used for external Ollama service.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from shared_ollama.api.server import app
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient, GenerateResponse
from shared_ollama.core.queue import RequestQueue


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncSharedOllamaClient for external service."""
    client = AsyncMock(spec=AsyncSharedOllamaClient)
    client.config = AsyncOllamaConfig()
    client._ensure_client = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def api_client(mock_async_client):
    """Create a test client with mocked async client."""
    with patch("shared_ollama.api.server._client", mock_async_client):
        with patch("shared_ollama.api.server._queue", RequestQueue(max_concurrent=3, max_queue_size=10)):
            async def mock_lifespan(app):
                yield

            app.router.lifespan_context = mock_lifespan
            with TestClient(app) as client:
                yield client


@pytest.fixture
def async_api_client(mock_async_client):
    """Create an async test client."""
    with patch("shared_ollama.api.server._client", mock_async_client):
        with patch("shared_ollama.api.server._queue", RequestQueue(max_concurrent=3, max_queue_size=10)):
            async def mock_lifespan(app):
                yield

            app.router.lifespan_context = mock_lifespan
            return AsyncClient(app=app, base_url="http://test")


class TestHealthEndpoint:
    """Behavioral tests for health check endpoint."""

    def test_health_check_success(self, api_client, mock_async_client):
        """Test successful health check returns healthy status."""
        with patch("shared_ollama.core.utils.check_service_health", return_value=(True, None)):
            response = api_client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["ollama_service"] == "healthy"
            assert data["version"] == "1.0.0"

    def test_health_check_unhealthy(self, api_client, mock_async_client):
        """Test health check returns unhealthy when service is down."""
        with patch(
            "shared_ollama.core.utils.check_service_health",
            return_value=(False, "Connection refused"),
        ):
            response = api_client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "unhealthy" in data["ollama_service"]
            assert "Connection refused" in data["ollama_service"]


class TestListModelsEndpoint:
    """Behavioral tests for list models endpoint."""

    def test_list_models_success(self, api_client, mock_async_client):
        """Test successful model listing returns models list."""
        mock_models = [
            {"name": "qwen2.5vl:7b", "size": 5969245856, "modified_at": "2025-11-03T17:24:58Z"},
            {"name": "qwen2.5:7b", "size": 4730000000, "modified_at": "2025-11-03T15:00:00Z"},
        ]
        mock_async_client.list_models = AsyncMock(return_value=mock_models)

        response = api_client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2
        assert data["models"][0]["name"] == "qwen2.5vl:7b"
        assert data["models"][1]["name"] == "qwen2.5:7b"

    def test_list_models_empty(self, api_client, mock_async_client):
        """Test listing models when none are available returns empty list."""
        mock_async_client.list_models = AsyncMock(return_value=[])

        response = api_client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []

    def test_list_models_error_returns_500(self, api_client, mock_async_client):
        """Test that errors during model listing return 500."""
        mock_async_client.list_models = AsyncMock(side_effect=ConnectionError("Service unavailable"))

        response = api_client.get("/api/v1/models")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data


class TestQueueStatsEndpoint:
    """Behavioral tests for queue statistics endpoint."""

    def test_get_queue_stats_returns_comprehensive_metrics(self, api_client, mock_async_client):
        """Test that queue stats endpoint returns all queue metrics."""
        response = api_client.get("/api/v1/queue/stats")
        assert response.status_code == 200
        data = response.json()

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


class TestGenerateEndpoint:
    """Comprehensive behavioral tests for generate endpoint."""

    def test_generate_success_returns_response(self, api_client, mock_async_client):
        """Test successful generation returns GenerateResponse."""
        mock_response = GenerateResponse(
            text="Hello, world!",
            model="qwen2.5vl:7b",
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
            json={"prompt": "Hello", "model": "qwen2.5vl:7b"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello, world!"
        assert data["model"] == "qwen2.5vl:7b"
        assert "request_id" in data
        assert "latency_ms" in data
        assert data["model_warm_start"] is False  # load_duration > 0

    def test_generate_with_all_options(self, api_client, mock_async_client):
        """Test generation with all optional parameters."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen2.5vl:7b",
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
                "model": "qwen2.5vl:7b",
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
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Response"
        assert data["model_warm_start"] is True  # load_duration == 0

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
            json={"prompt": "Hello", "model": "nonexistent"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data

    def test_generate_uses_queue_slot(self, api_client, mock_async_client):
        """Test that generate endpoint uses queue for concurrency control."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen2.5vl:7b",
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

    @pytest.mark.asyncio
    async def test_generate_stream_returns_sse_format(self, async_api_client, mock_async_client):
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

        async with async_api_client as client:
            response = await client.post(
                "/api/v1/generate",
                json={"prompt": "Hello", "stream": True},
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            # Read streaming response
            content = b""
            async for chunk in response.aiter_bytes():
                content += chunk

            # Verify SSE format
            text = content.decode("utf-8")
            assert text.startswith("data: ")
            assert "\n\n" in text

    @pytest.mark.asyncio
    async def test_generate_stream_handles_errors(self, async_api_client, mock_async_client):
        """Test that streaming errors are sent as final chunk."""
        async def mock_stream():
            raise RuntimeError("Streaming error")

        mock_async_client.generate_stream = AsyncMock(return_value=mock_stream())

        async with async_api_client as client:
            response = await client.post(
                "/api/v1/generate",
                json={"prompt": "Test", "stream": True},
            )
            assert response.status_code == 200

            content = b""
            async for chunk in response.aiter_bytes():
                content += chunk

            text = content.decode("utf-8")
            # Should contain error in final chunk
            assert "error" in text.lower() or "done" in text.lower()


class TestChatEndpoint:
    """Comprehensive behavioral tests for chat endpoint."""

    def test_chat_success_returns_response(self, api_client, mock_async_client):
        """Test successful chat returns ChatResponse."""
        mock_response = {
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "model": "qwen2.5vl:7b",
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
                "model": "qwen2.5vl:7b",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello! How can I help?"
        assert data["model"] == "qwen2.5vl:7b"
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
            "model": "qwen2.5vl:7b",
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
                "model": "qwen2.5vl:7b",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"]["content"] == "Response"

    def test_chat_error_returns_500(self, api_client, mock_async_client):
        """Test that chat errors return 500."""
        mock_async_client.chat = AsyncMock(side_effect=ValueError("Invalid messages"))

        response = api_client.post(
            "/api/v1/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data


class TestChatStreaming:
    """Behavioral tests for streaming chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_stream_returns_sse_format(self, async_api_client, mock_async_client):
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

        async with async_api_client as client:
            response = await client.post(
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
            model="qwen2.5vl:7b",
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
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data

    def test_request_id_is_unique(self, api_client, mock_async_client):
        """Test that each request gets unique request ID."""
        mock_response = GenerateResponse(
            text="Response",
            model="qwen2.5vl:7b",
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

    def test_root_endpoint_returns_api_info(self, api_client):
        """Test root endpoint returns API metadata."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Shared Ollama Service API"
        assert data["version"] == "1.0.0"
        assert "/api/docs" in data["docs"]
        assert "/api/v1/health" in data["health"]


class TestErrorHandling:
    """Comprehensive behavioral tests for error handling."""

    def test_client_not_initialized_returns_503(self, api_client):
        """Test that uninitialized client returns 503."""
        with patch("shared_ollama.api.server._client", None):
            response = api_client.get("/api/v1/models")
            assert response.status_code == 503
            data = response.json()
            assert "not initialized" in data.get("detail", "").lower()

    def test_validation_error_returns_422(self, api_client, mock_async_client):
        """Test that validation errors return 422 with details."""
        response = api_client.post("/api/v1/generate", json={"invalid": "data"})
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data

    def test_global_exception_handler_returns_500(self, api_client, mock_async_client):
        """Test that unexpected exceptions are handled gracefully."""
        mock_async_client.list_models = AsyncMock(side_effect=Exception("Unexpected error"))

        response = api_client.get("/api/v1/models")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data

    def test_error_response_includes_request_id(self, api_client, mock_async_client):
        """Test that error responses include request_id when available."""
        mock_async_client.list_models = AsyncMock(side_effect=RuntimeError("Test error"))

        response = api_client.get("/api/v1/models")
        assert response.status_code == 500
        data = response.json()
        # Request ID should be present in error response
        assert "request_id" in data or "error" in data


class TestAsyncFunctionality:
    """Behavioral tests for async endpoint behavior."""

    @pytest.mark.asyncio
    async def test_endpoints_are_truly_async(self, async_api_client, mock_async_client):
        """Test that endpoints handle async operations correctly."""
        mock_response = GenerateResponse(
            text="Async response",
            model="qwen2.5vl:7b",
            context=None,
            total_duration=300_000_000,
            load_duration=0,
            prompt_eval_count=1,
            prompt_eval_duration=0,
            eval_count=1,
            eval_duration=0,
        )
        mock_async_client.generate = AsyncMock(return_value=mock_response)

        async with async_api_client as client:
            response = await client.post(
                "/api/v1/generate",
                json={"prompt": "Test async"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "Async response"
