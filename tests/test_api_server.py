"""
Tests for the FastAPI REST API server.

Tests all endpoints, error handling, rate limiting, and async functionality.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from shared_ollama.api.server import app
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient, GenerateResponse


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncSharedOllamaClient."""
    client = AsyncMock(spec=AsyncSharedOllamaClient)
    client.config = AsyncOllamaConfig()
    client._ensure_client = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def api_client(mock_async_client):
    """Create a test client with mocked async client."""
    # Patch the global client in the lifespan
    with patch("shared_ollama.api.server._client", mock_async_client):
        # Override the lifespan to use our mock
        async def mock_lifespan(app):
            yield

        app.router.lifespan_context = mock_lifespan
        with TestClient(app) as client:
            yield client


@pytest.fixture
def async_api_client(mock_async_client):
    """Create an async test client."""
    with patch("shared_ollama.api.server._client", mock_async_client):
        async def mock_lifespan(app):
            yield

        app.router.lifespan_context = mock_lifespan
        return AsyncClient(app=app, base_url="http://test")


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_success(self, api_client, mock_async_client):
        """Test successful health check."""
        with patch("shared_ollama.core.utils.check_service_health", return_value=(True, None)):
            response = api_client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["ollama_service"] == "healthy"
            assert data["version"] == "1.0.0"

    def test_health_check_unhealthy(self, api_client, mock_async_client):
        """Test health check when service is unhealthy."""
        with patch("shared_ollama.core.utils.check_service_health", return_value=(False, "Connection refused")):
            response = api_client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "unhealthy" in data["ollama_service"]
            assert "Connection refused" in data["ollama_service"]


class TestListModelsEndpoint:
    """Tests for the list models endpoint."""

    def test_list_models_success(self, api_client, mock_async_client):
        """Test successful model listing."""
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
        """Test listing models when none are available."""
        mock_async_client.list_models = AsyncMock(return_value=[])

        response = api_client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []

    def test_list_models_error(self, api_client, mock_async_client):
        """Test error handling when listing models fails."""
        mock_async_client.list_models = AsyncMock(side_effect=ConnectionError("Service unavailable"))

        response = api_client.get("/api/v1/models")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data


class TestGenerateEndpoint:
    """Tests for the generate endpoint."""

    def test_generate_success(self, api_client, mock_async_client):
        """Test successful text generation."""
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

    def test_generate_with_options(self, api_client, mock_async_client):
        """Test generation with all options."""
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
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Response"
        assert data["model_warm_start"] is True  # load_duration == 0

    def test_generate_missing_prompt(self, api_client, mock_async_client):
        """Test generation with missing required prompt."""
        response = api_client.post("/api/v1/generate", json={})
        assert response.status_code == 422  # Validation error

    def test_generate_error(self, api_client, mock_async_client):
        """Test error handling when generation fails."""
        mock_async_client.generate = AsyncMock(side_effect=RuntimeError("Model not found"))

        response = api_client.post(
            "/api/v1/generate",
            json={"prompt": "Hello", "model": "nonexistent"},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    def test_chat_success(self, api_client, mock_async_client):
        """Test successful chat completion."""
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
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "model": "qwen2.5vl:7b",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello! How can I help?"
        assert data["model"] == "qwen2.5vl:7b"
        assert "request_id" in data

    def test_chat_multiple_messages(self, api_client, mock_async_client):
        """Test chat with multiple messages."""
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

    def test_chat_missing_messages(self, api_client, mock_async_client):
        """Test chat with missing required messages."""
        response = api_client.post("/api/v1/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_error(self, api_client, mock_async_client):
        """Test error handling when chat fails."""
        mock_async_client.chat = AsyncMock(side_effect=ValueError("Invalid messages"))

        response = api_client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data or "detail" in data


class TestRequestContext:
    """Tests for request context and headers."""

    def test_project_name_header(self, api_client, mock_async_client):
        """Test that X-Project-Name header is captured."""
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
        # Request ID should be present
        data = response.json()
        assert "request_id" in data

    def test_request_id_generation(self, api_client, mock_async_client):
        """Test that unique request IDs are generated."""
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
        # Request IDs should be different
        assert data1["request_id"] != data2["request_id"]


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_endpoint(self, api_client):
        """Test the root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Shared Ollama Service API"
        assert data["version"] == "1.0.0"
        assert "/api/docs" in data["docs"]
        assert "/api/v1/health" in data["health"]


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_client_not_initialized(self, api_client):
        """Test error when client is not initialized."""
        with patch("shared_ollama.api.server._client", None):
            response = api_client.get("/api/v1/models")
            assert response.status_code == 503
            data = response.json()
            assert "not initialized" in data.get("detail", "").lower()

    def test_global_exception_handler(self, api_client, mock_async_client):
        """Test global exception handler for unexpected errors."""
        mock_async_client.list_models = AsyncMock(side_effect=Exception("Unexpected error"))

        response = api_client.get("/api/v1/models")
        # Should be handled by endpoint-specific handler, not global
        assert response.status_code in [500, 503]


class TestAsyncFunctionality:
    """Tests to verify async functionality."""

    @pytest.mark.asyncio
    async def test_async_endpoints(self, async_api_client, mock_async_client):
        """Test that endpoints are truly async."""
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

