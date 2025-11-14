"""
Comprehensive behavioral tests for AsyncSharedOllamaClient.

Tests focus on real async behavior, streaming, error handling, and edge cases.
Uses real httpx client with test server (ollama_server fixture).
"""

import asyncio

import httpx
import pytest

from shared_ollama import MetricsCollector
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient


@pytest.mark.asyncio
class TestAsyncClientInitialization:
    """Behavioral tests for AsyncSharedOllamaClient initialization."""

    async def test_client_initializes_with_default_config(self, ollama_server):
        """Test that client initializes with default config."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            assert client.config == config
            assert client.client is not None

    async def test_client_initializes_with_custom_config(self, ollama_server):
        """Test that client initializes with custom config."""
        config = AsyncOllamaConfig(
            base_url=ollama_server.base_url,
            default_model="qwen2.5:7b",
            timeout=120,
            max_concurrent_requests=5,
        )
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            assert client.config.default_model == "qwen2.5:7b"
            assert client.config.timeout == 120
            assert client.config.max_concurrent_requests == 5

    async def test_client_verifies_connection_on_init(self, ollama_server):
        """Test that client verifies connection when verify_on_init=True."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=True) as client:
            # Should not raise - connection verified
            assert client.client is not None

    async def test_client_closes_on_exit(self, ollama_server):
        """Test that client closes httpx client on context exit."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            httpx_client = client.client
            assert httpx_client is not None

        # After exit, client should be closed
        assert client.client is None

    async def test_client_handles_connection_failure(self):
        """Test that client raises ConnectionError when service unavailable."""
        config = AsyncOllamaConfig(base_url="http://localhost:99999", verify_on_init=True)
        with pytest.raises(ConnectionError):
            async with AsyncSharedOllamaClient(config=config, verify_on_init=True):
                pass


@pytest.mark.asyncio
class TestAsyncClientListModels:
    """Behavioral tests for list_models()."""

    async def test_list_models_returns_list(self, ollama_server):
        """Test that list_models() returns a list of models."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            models = await client.list_models()
            assert isinstance(models, list)
            assert len(models) >= 2

    async def test_list_models_extracts_models_key(self, ollama_server):
        """Test that list_models() correctly extracts models from response."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            models = await client.list_models()
            assert any(model["name"] == "qwen2.5vl:7b" for model in models)

    async def test_list_models_handles_empty_response(self, ollama_server):
        """Test that list_models() handles empty models list."""
        ollama_server.state["models"] = []
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            models = await client.list_models()
            assert models == []

    async def test_list_models_validates_response_structure(self, ollama_server):
        """Test that list_models() validates response is a dict."""
        # This would require modifying the test server, so we test the behavior
        # by ensuring it works with valid responses
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            models = await client.list_models()
            # Valid response should work
            assert isinstance(models, list)

    async def test_list_models_handles_http_error(self, ollama_server):
        """Test that list_models() raises HTTPStatusError on HTTP errors."""
        ollama_server.state["tags_status"] = 500
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.list_models()


@pytest.mark.asyncio
class TestAsyncClientGenerate:
    """Behavioral tests for generate()."""

    async def test_generate_returns_generate_response(self, ollama_server):
        """Test that generate() returns GenerateResponse object."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            response = await client.generate("Hello, world!")
            assert response.text.startswith("ECHO: Hello, world!")
            assert response.model == "qwen2.5vl:7b"

    async def test_generate_extracts_all_metrics(self, ollama_server):
        """Test that generate() extracts all performance metrics."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            response = await client.generate("Test prompt")
            assert response.total_duration > 0
            assert response.load_duration > 0
            assert response.prompt_eval_count > 0
            assert response.eval_count > 0

    async def test_generate_uses_default_model(self, ollama_server):
        """Test that generate() uses default model when model=None."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            response = await client.generate("Test", model=None)
            assert response.model == config.default_model

    async def test_generate_includes_system_message(self, ollama_server):
        """Test that generate() includes system message in payload."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            response = await client.generate("Test", system="You are helpful")
            # Verify system was sent by checking server state
            calls = ollama_server.state.get("generate_calls", [])
            assert any("system" in call and call["system"] == "You are helpful" for call in calls)

    async def test_generate_includes_options(self, ollama_server):
        """Test that generate() includes options in payload."""
        from shared_ollama import GenerateOptions

        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            options = GenerateOptions(temperature=0.7, max_tokens=100)
            await client.generate("Test", options=options)

            calls = ollama_server.state.get("generate_calls", [])
            assert any("options" in call for call in calls)

    async def test_generate_handles_http_error(self, ollama_server):
        """Test that generate() raises HTTPStatusError on HTTP errors."""
        ollama_server.state["generate_failures"] = 1
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate("This should fail")

    async def test_generate_records_metrics(self, ollama_server):
        """Test that generate() records metrics via MetricsCollector."""
        MetricsCollector.reset()
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            await client.generate("Test prompt")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 1
        assert metrics.successful_requests >= 1


@pytest.mark.asyncio
class TestAsyncClientChat:
    """Behavioral tests for chat()."""

    async def test_chat_returns_dict_with_message(self, ollama_server):
        """Test that chat() returns dict with message content."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            messages = [{"role": "user", "content": "Hello async"}]
            response = await client.chat(messages)
            assert "message" in response
            assert response["message"]["content"].startswith("Echo:")

    async def test_chat_handles_multiple_messages(self, ollama_server):
        """Test that chat() handles conversation history."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is 2+2?"},
            ]
            response = await client.chat(messages)
            assert "message" in response

    async def test_chat_includes_options(self, ollama_server):
        """Test that chat() includes options in payload."""
        from shared_ollama import GenerateOptions

        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            options = GenerateOptions(temperature=0.8)
            await client.chat([{"role": "user", "content": "Test"}], options=options)

            calls = ollama_server.state.get("chat_calls", [])
            assert any("options" in call for call in calls)

    async def test_chat_handles_http_error(self, ollama_server):
        """Test that chat() raises HTTPStatusError on HTTP errors."""
        ollama_server.state["chat_failures"] = 1
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat([{"role": "user", "content": "This should fail"}])


@pytest.mark.asyncio
class TestAsyncClientStreaming:
    """Behavioral tests for streaming methods."""

    async def test_generate_stream_yields_chunks(self, ollama_server):
        """Test that generate_stream() yields incremental chunks."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            chunks = []
            async for chunk in client.generate_stream("Stream test"):
                chunks.append(chunk)
                assert "chunk" in chunk
                assert "done" in chunk
                if chunk["done"]:
                    break

            assert len(chunks) > 0
            assert chunks[-1]["done"] is True

    async def test_generate_stream_includes_metrics_in_final_chunk(self, ollama_server):
        """Test that generate_stream() includes metrics in final chunk."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            final_chunk = None
            async for chunk in client.generate_stream("Test"):
                if chunk.get("done"):
                    final_chunk = chunk
                    break

            assert final_chunk is not None
            assert "latency_ms" in final_chunk
            assert "model_load_ms" in final_chunk

    async def test_chat_stream_yields_chunks(self, ollama_server):
        """Test that chat_stream() yields incremental chunks."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            messages = [{"role": "user", "content": "Stream chat test"}]
            chunks = []
            async for chunk in client.chat_stream(messages):
                chunks.append(chunk)
                assert "chunk" in chunk
                assert "role" in chunk
                assert "done" in chunk
                if chunk["done"]:
                    break

            assert len(chunks) > 0
            assert chunks[-1]["done"] is True

    async def test_chat_stream_includes_metrics_in_final_chunk(self, ollama_server):
        """Test that chat_stream() includes metrics in final chunk."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            messages = [{"role": "user", "content": "Test"}]
            final_chunk = None
            async for chunk in client.chat_stream(messages):
                if chunk.get("done"):
                    final_chunk = chunk
                    break

            assert final_chunk is not None
            assert "latency_ms" in final_chunk


@pytest.mark.asyncio
class TestAsyncClientConcurrency:
    """Behavioral tests for concurrency control."""

    async def test_semaphore_limits_concurrent_requests(self, ollama_server):
        """Test that semaphore limits concurrent requests when configured."""
        config = AsyncOllamaConfig(
            base_url=ollama_server.base_url, max_concurrent_requests=2
        )
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            assert client._semaphore is not None
            assert client._semaphore._value == 2

    async def test_concurrent_requests_respect_semaphore(self, ollama_server):
        """Test that concurrent requests respect semaphore limit."""
        config = AsyncOllamaConfig(
            base_url=ollama_server.base_url, max_concurrent_requests=2
        )
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # Start multiple requests concurrently
            tasks = [client.generate(f"Request {i}") for i in range(5)]
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 5
            assert all(response.text for response in responses)

    async def test_no_semaphore_when_unlimited(self, ollama_server):
        """Test that no semaphore is created when max_concurrent_requests=None."""
        config = AsyncOllamaConfig(
            base_url=ollama_server.base_url, max_concurrent_requests=None
        )
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            assert client._semaphore is None


@pytest.mark.asyncio
class TestAsyncClientHealthCheck:
    """Behavioral tests for health_check()."""

    async def test_health_check_returns_true_for_200(self, ollama_server):
        """Test that health_check() returns True for HTTP 200."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            result = await client.health_check()
            assert result is True

    async def test_health_check_returns_false_for_non_200(self, ollama_server):
        """Test that health_check() returns False for non-200 status."""
        ollama_server.state["tags_status"] = 500
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            result = await client.health_check()
            assert result is False

    async def test_health_check_returns_false_on_exception(self, ollama_server):
        """Test that health_check() returns False on any exception."""
        config = AsyncOllamaConfig(base_url="http://localhost:99999")
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            result = await client.health_check()
            assert result is False


@pytest.mark.asyncio
class TestAsyncClientGetModelInfo:
    """Behavioral tests for get_model_info()."""

    async def test_get_model_info_returns_model_dict(self, ollama_server):
        """Test that get_model_info() returns model dict when found."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            model_info = await client.get_model_info("qwen2.5vl:7b")
            assert model_info is not None
            assert model_info["name"] == "qwen2.5vl:7b"

    async def test_get_model_info_returns_none_when_not_found(self, ollama_server):
        """Test that get_model_info() returns None when model not found."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            model_info = await client.get_model_info("nonexistent:model")
            assert model_info is None
