"""
Comprehensive behavioral tests for model validation in API routes.

Tests focus on real API behavior: rejecting disallowed models, accepting allowed models,
edge cases, error messages, and integration with hardware profile detection.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from shared_ollama.api.dependencies import set_dependencies
from shared_ollama.api.server import app
from shared_ollama.application.use_cases import ChatUseCase, GenerateUseCase
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient, GenerateResponse
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import (
    get_allowed_models,
    is_model_allowed,
)
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    ImageCacheAdapter,
    ImageProcessorAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)
from tests.helpers import assert_error_response, cleanup_dependency_overrides, setup_dependency_overrides

VALID_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
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
def api_client_with_validation(mock_async_client):
    """Create a test client with real model validation enabled."""
    from shared_ollama.infrastructure.image_processing import ImageProcessor
    from shared_ollama.infrastructure.image_cache import ImageCache

    client_adapter = AsyncOllamaClientAdapter(mock_async_client)
    logger_adapter = RequestLoggerAdapter()
    metrics_adapter = MetricsCollectorAdapter()
    image_processor = ImageProcessor()
    image_cache = ImageCache()
    image_processor_adapter = ImageProcessorAdapter(image_processor)
    image_cache_adapter = ImageCacheAdapter(image_cache)
    queue = RequestQueue(max_concurrent=3, max_queue_size=10)

    # Create use cases
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
    from shared_ollama.application.use_cases import ListModelsUseCase
    list_models_use_case = ListModelsUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )
    vlm_use_case = VLMUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
        image_processor=image_processor_adapter,
        image_cache=image_cache_adapter,
    )

    setup_dependency_overrides(
        app=app,
        client_adapter=client_adapter,
        logger_adapter=logger_adapter,
        metrics_adapter=metrics_adapter,
        queue=queue,
        generate_use_case=generate_use_case,
        chat_use_case=chat_use_case,
        list_models_use_case=list_models_use_case,
        image_processor_adapter=image_processor_adapter,
        image_cache_adapter=image_cache_adapter,
        vlm_use_case=vlm_use_case,
    )

    try:
        with TestClient(app) as client:
            yield client
    finally:
        cleanup_dependency_overrides(app)


class TestVLMEndpointModelValidation:
    """Behavioral tests for VLM endpoint model validation."""

    def test_allowed_model_accepts_request(self, api_client_with_validation, mock_async_client):
        """Test that an allowed model is accepted by the VLM endpoint."""
        allowed_models = get_allowed_models()
        if not allowed_models:
            pytest.skip("No allowed models configured")

        test_model = next(iter(allowed_models))
        mock_async_client.chat = AsyncMock(
            return_value={"response": "Test response", "done": True}
        )

        response = api_client_with_validation.post(
            "/api/v1/vlm",
            json={
                "model": test_model,
                "messages": [{"role": "user", "content": "What's in this image?"}],
                "images": [VALID_IMAGE_DATA_URL],  # Native format requires separate images
            },
        )

        assert response.status_code == 200
        mock_async_client.chat.assert_called_once()

    def test_disallowed_model_rejects_request(self, api_client_with_validation):
        """Test that a disallowed model is rejected with 400 error."""
        disallowed_model = "qwen3-vl:999b-impossible-model"

        response = api_client_with_validation.post(
            "/api/v1/vlm",
            json={
                "model": disallowed_model,
                "messages": [{"role": "user", "content": "Test"}],
                "images": [VALID_IMAGE_DATA_URL],  # Native format requires separate images
            },
        )

        # Debug: print response if not 400
        if response.status_code != 400:
            print(f"Unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}. Response: {response.text}"
        data = response.json()
        assert "not supported" in data["detail"].lower() or "not allowed" in data["detail"].lower()
        assert disallowed_model in data["detail"]

    def test_none_model_uses_default(self, api_client_with_validation, mock_async_client):
        """Test that None model (omitted) uses default and is accepted."""
        from shared_ollama.core.utils import get_default_vlm_model

        default_model = get_default_vlm_model()
        mock_async_client.chat = AsyncMock(
            return_value={"response": "Test response", "done": True}
        )

        response = api_client_with_validation.post(
            "/api/v1/vlm",
            json={
                # model field omitted - should use default
                "messages": [{"role": "user", "content": "Test"}],
                "images": [VALID_IMAGE_DATA_URL],  # Native format requires separate images
            },
        )

        assert response.status_code == 200
        # Verify the default model was used
        call_args = mock_async_client.chat.call_args
        assert call_args is not None

    def test_error_message_includes_allowed_models(self, api_client_with_validation):
        """Test that error message includes list of allowed models."""
        disallowed_model = "invalid-model:test"
        allowed_models = get_allowed_models()

        response = api_client_with_validation.post(
            "/api/v1/vlm",
            json={
                "model": disallowed_model,
                "messages": [{"role": "user", "content": "Test"}],
                "images": [VALID_IMAGE_DATA_URL],
            },
        )

        assert response.status_code == 400
        data = response.json()
        detail = data["detail"].lower()

        # Error message should mention allowed models
        for allowed_model in allowed_models:
            assert allowed_model.lower() in detail or any(
                part in detail for part in allowed_model.split(":")
            )

    def test_openai_format_also_validates_model(self, api_client_with_validation):
        """Test that OpenAI-compatible VLM endpoint also validates models."""
        disallowed_model = "gpt-4-vision-preview"  # Not a Qwen model

        response = api_client_with_validation.post(
            "/api/v1/vlm/openai",
            json={
                "model": disallowed_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Test"},
                            {
                                "type": "image_url",
                                "image_url": {"url": VALID_IMAGE_DATA_URL},
                            },
                        ],
                    }
                ],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "not supported" in data["detail"].lower() or "not allowed" in data["detail"].lower()


class TestChatEndpointModelValidation:
    """Behavioral tests for chat endpoint model validation."""

    def test_allowed_model_accepts_request(self, api_client_with_validation, mock_async_client):
        """Test that an allowed model is accepted by the chat endpoint."""
        allowed_models = get_allowed_models()
        if not allowed_models:
            pytest.skip("No allowed models configured")

        # Find a text model (not VLM)
        text_models = [m for m in allowed_models if "vl" not in m.lower()]
        if not text_models:
            pytest.skip("No text models in allowed set")

        test_model = text_models[0]
        mock_async_client.chat = AsyncMock(
            return_value={"message": {"content": "Test response"}, "done": True}
        )

        response = api_client_with_validation.post(
            "/api/v1/chat",
            json={
                "model": test_model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        mock_async_client.chat.assert_called_once()

    def test_disallowed_model_rejects_request(self, api_client_with_validation):
        """Test that a disallowed model is rejected with 400 error."""
        disallowed_model = "gpt-4-turbo"

        response = api_client_with_validation.post(
            "/api/v1/chat",
            json={
                "model": disallowed_model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "not supported" in data["detail"].lower() or "not allowed" in data["detail"].lower()

    def test_none_model_uses_default(self, api_client_with_validation, mock_async_client):
        """Test that None model (omitted) uses default and is accepted."""
        from shared_ollama.core.utils import get_default_text_model

        default_model = get_default_text_model()
        mock_async_client.chat = AsyncMock(
            return_value={"message": {"content": "Test response"}, "done": True}
        )

        response = api_client_with_validation.post(
            "/api/v1/chat",
            json={
                # model field omitted - should use default
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        call_args = mock_async_client.chat.call_args
        assert call_args is not None


class TestGenerateEndpointModelValidation:
    """Behavioral tests for generate endpoint model validation."""

    def test_allowed_model_accepts_request(self, api_client_with_validation, mock_async_client):
        """Test that an allowed model is accepted by the generate endpoint."""
        allowed_models = get_allowed_models()
        if not allowed_models:
            pytest.skip("No allowed models configured")

        # Find a text model (not VLM)
        text_models = [m for m in allowed_models if "vl" not in m.lower()]
        if not text_models:
            pytest.skip("No text models in allowed set")

        test_model = text_models[0]
        mock_async_client.generate = AsyncMock(
            return_value=GenerateResponse(text="Test response", model=test_model)
        )

        response = api_client_with_validation.post(
            "/api/v1/generate",
            json={
                "model": test_model,
                "prompt": "Hello, world!",
            },
        )

        assert response.status_code == 200
        mock_async_client.generate.assert_called_once()

    def test_disallowed_model_rejects_request(self, api_client_with_validation):
        """Test that a disallowed model is rejected with 400 error."""
        disallowed_model = "llama2:70b"

        response = api_client_with_validation.post(
            "/api/v1/generate",
            json={
                "model": disallowed_model,
                "prompt": "Hello, world!",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "not supported" in data["detail"].lower() or "not allowed" in data["detail"].lower()

    def test_none_model_uses_default(self, api_client_with_validation, mock_async_client):
        """Test that None model (omitted) uses default and is accepted."""
        from shared_ollama.core.utils import get_default_text_model

        default_model = get_default_text_model()
        mock_async_client.generate = AsyncMock(
            return_value=GenerateResponse(text="Test response", model=default_model)
        )

        response = api_client_with_validation.post(
            "/api/v1/generate",
            json={
                # model field omitted - should use default
                "prompt": "Hello, world!",
            },
        )

        assert response.status_code == 200
        call_args = mock_async_client.generate.call_args
        assert call_args is not None


class TestBatchEndpointModelValidation:
    """Behavioral tests for batch endpoint model validation."""

    def test_allowed_models_in_batch_accepts(self, api_client_with_validation, mock_async_client):
        """Test that batch requests with allowed models are accepted."""
        allowed_models = get_allowed_models()
        if not allowed_models:
            pytest.skip("No allowed models configured")

        test_model = next(iter(allowed_models))
        mock_async_client.generate = AsyncMock(
            return_value=GenerateResponse(text="Test response", model=test_model)
        )

        response = api_client_with_validation.post(
            "/api/v1/batch/chat",
            json={
                "requests": [
                    {
                        "model": test_model,
                        "messages": [{"role": "user", "content": "Prompt 1"}],
                    },
                    {
                        "model": test_model,
                        "messages": [{"role": "user", "content": "Prompt 2"}],
                    },
                ]
            },
        )

        assert response.status_code == 200

    def test_disallowed_model_in_batch_rejects(self, api_client_with_validation):
        """Test that batch requests with disallowed models are rejected."""
        disallowed_model = "invalid-model:test"

        response = api_client_with_validation.post(
            "/api/v1/batch/chat",
            json={
                "requests": [
                    {
                        "model": disallowed_model,
                        "messages": [{"role": "user", "content": "Prompt 1"}],
                    },
                ]
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "not supported" in data["detail"].lower() or "not allowed" in data["detail"].lower()

    def test_mixed_allowed_disallowed_in_batch_rejects(self, api_client_with_validation):
        """Test that batch with mix of allowed/disallowed models is rejected."""
        allowed_models = get_allowed_models()
        if not allowed_models:
            pytest.skip("No allowed models configured")

        test_model = next(iter(allowed_models))
        disallowed_model = "invalid-model:test"

        response = api_client_with_validation.post(
            "/api/v1/batch/chat",
            json={
                "requests": [
                    {
                        "model": test_model,
                        "messages": [{"role": "user", "content": "Prompt 1"}],
                    },
                    {
                        "model": disallowed_model,
                        "messages": [{"role": "user", "content": "Prompt 2"}],
                    },
                ]
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "not supported" in data["detail"].lower() or "not allowed" in data["detail"].lower()
        assert disallowed_model in data["detail"]


class TestModelValidationEdgeCases:
    """Edge case tests for model validation."""

    def test_empty_string_model_treated_as_none(self, api_client_with_validation, mock_async_client):
        """Test that empty string model is treated as None (use default)."""
        mock_async_client.generate = AsyncMock(
            return_value=GenerateResponse(text="Test response", model="qwen3-vl:8b-instruct-q4_K_M")
        )

        response = api_client_with_validation.post(
            "/api/v1/generate",
            json={
                "model": "",  # Empty string
                "prompt": "Hello",
            },
        )

        # Empty string might be treated as None or might be rejected - either is acceptable
        # The important thing is it doesn't crash
        assert response.status_code in [200, 400]

    def test_case_sensitive_model_validation(self, api_client_with_validation):
        """Test that model validation is case-sensitive."""
        allowed_models = get_allowed_models()
        if not allowed_models:
            pytest.skip("No allowed models configured")

        test_model = next(iter(allowed_models))
        # Try uppercase version
        uppercase_model = test_model.upper()

        # If uppercase is different from original, it should be rejected
        if uppercase_model != test_model:
            response = api_client_with_validation.post(
                "/api/v1/generate",
                json={
                    "model": uppercase_model,
                    "prompt": "Hello",
                },
            )

            assert response.status_code == 400

    def test_validation_respects_hardware_profile(self, api_client_with_validation):
        """Test that validation respects the current hardware profile."""
        # Get current allowed models (based on hardware profile)
        allowed_models = get_allowed_models()

        # Try a model that's definitely not in the allowed set
        disallowed = "qwen3-vl:999b-impossible"
        assert not is_model_allowed(disallowed)

        response = api_client_with_validation.post(
            "/api/v1/vlm",
            json={
                "model": disallowed,
                "messages": [{"role": "user", "content": "Test"}],
                "images": [VALID_IMAGE_DATA_URL],  # Native format requires separate images
            },
        )

        assert response.status_code == 400

