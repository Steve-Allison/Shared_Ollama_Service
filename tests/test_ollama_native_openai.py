"""
Tests for Ollama's native OpenAI compatibility endpoints.

These tests validate that Ollama's native /v1/chat/completions and /v1/embeddings
endpoints work correctly, which is important for understanding what Ollama provides
out of the box vs. what the Shared Ollama Service adds.
"""

import pytest
import httpx


@pytest.mark.integration
class TestOllamaNativeOpenAI:
    """Tests for Ollama's native OpenAI-compatible endpoints."""

    @pytest.fixture
    def ollama_base_url(self):
        """Base URL for Ollama service."""
        return "http://localhost:11434"

    def test_native_chat_completions_endpoint_exists(self, ollama_base_url):
        """Test that Ollama's native /v1/chat/completions endpoint exists."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{ollama_base_url}/v1/chat/completions",
                json={
                    "model": "qwen3:14b-q4_K_M",
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
            )
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert "choices" in data, "Response should have 'choices' field (OpenAI format)"
            assert len(data["choices"]) > 0, "Response should have at least one choice"
            assert "message" in data["choices"][0], "Choice should have 'message' field"
            assert "content" in data["choices"][0]["message"], "Message should have 'content' field"

    def test_native_chat_completions_response_format(self, ollama_base_url):
        """Test that native endpoint returns OpenAI-compatible format."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{ollama_base_url}/v1/chat/completions",
                json={
                    "model": "qwen3:14b-q4_K_M",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            
            # Check OpenAI-compatible fields
            assert "id" in data, "Should have 'id' field"
            assert "object" in data, "Should have 'object' field"
            assert data["object"] == "chat.completion", "Object should be 'chat.completion'"
            assert "created" in data, "Should have 'created' timestamp"
            assert "model" in data, "Should have 'model' field"
            assert "choices" in data, "Should have 'choices' array"
            assert "usage" in data, "Should have 'usage' field"
            
            # Check usage fields
            assert "prompt_tokens" in data["usage"], "Should have prompt_tokens"
            assert "completion_tokens" in data["usage"], "Should have completion_tokens"
            assert "total_tokens" in data["usage"], "Should have total_tokens"

    def test_native_embeddings_endpoint_exists(self, ollama_base_url):
        """Test that Ollama's native /v1/embeddings endpoint exists."""
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{ollama_base_url}/v1/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "input": "test embedding",
                },
            )
            # May return 404 if embeddings model not available, but endpoint should exist
            assert response.status_code in (200, 404), f"Unexpected status: {response.status_code}"
            if response.status_code == 200:
                data = response.json()
                assert "data" in data, "Response should have 'data' field (OpenAI format)"
                assert len(data["data"]) > 0, "Response should have at least one embedding"
                assert "embedding" in data["data"][0], "Embedding should have 'embedding' field"

    def test_native_vs_wrapper_difference(self, ollama_base_url):
        """Test to demonstrate what native Ollama provides vs wrapper."""
        # Native Ollama provides basic OpenAI compatibility
        with httpx.Client(timeout=10.0) as client:
            native_response = client.post(
                f"{ollama_base_url}/v1/chat/completions",
                json={
                    "model": "qwen3:14b-q4_K_M",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
            assert native_response.status_code == 200
            native_data = native_response.json()
            assert "choices" in native_data, "Native endpoint should work"
            
        # Note: This test documents that native Ollama works
        # The wrapper adds: queuing, rate limiting, image processing, observability, etc.

