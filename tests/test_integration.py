"""
Integration tests for end-to-end workflows.

Tests focus on real end-to-end flows, full request/response cycles,
multi-step workflows, and real service interactions.
Uses real server (ollama_server fixture) - no mocks of internal logic.
"""

import asyncio

import pytest

from shared_ollama import (
    AsyncOllamaConfig,
    AsyncSharedOllamaClient,
    GenerateOptions,
    MetricsCollector,
    SharedOllamaClient,
)


@pytest.mark.asyncio
class TestEndToEndGeneration:
    """End-to-end tests for text generation workflow."""

    async def test_full_generation_workflow(self, ollama_server):
        """Test complete generation workflow from request to response."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            response = await client.generate("Hello, world!")

            assert response.text.startswith("ECHO: Hello, world!")
            assert response.model == "qwen2.5vl:7b"
            assert response.total_duration > 0
            assert response.eval_count > 0

    async def test_generation_with_all_options(self, ollama_server):
        """Test generation with all options configured."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            options = GenerateOptions(
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                max_tokens=100,
                seed=42,
                stop=["\n"],
            )

            response = await client.generate(
                "Generate a story",
                model="qwen2.5vl:7b",
                system="You are a creative writer",
                options=options,
                format="json",
            )

            assert isinstance(response, GenerateResponse)
            assert response.model == "qwen2.5vl:7b"

    async def test_generation_workflow_with_metrics(self, ollama_server):
        """Test that generation workflow records metrics correctly."""
        MetricsCollector.reset()
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            await client.generate("Test prompt 1")
            await client.generate("Test prompt 2")
            await client.generate("Test prompt 3")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 3
        assert metrics.successful_requests >= 3
        assert metrics.requests_by_operation["generate"] >= 3


@pytest.mark.asyncio
class TestEndToEndChat:
    """End-to-end tests for chat completion workflow."""

    async def test_full_chat_workflow(self, ollama_server):
        """Test complete chat workflow with conversation history."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # First message
            messages1 = [{"role": "user", "content": "What is 2+2?"}]
            response1 = await client.chat(messages1)

            assert "message" in response1
            assert response1["message"]["content"].startswith("Echo:")

            # Follow-up message
            messages2 = [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": response1["message"]["content"]},
                {"role": "user", "content": "What about 3+3?"},
            ]
            response2 = await client.chat(messages2)

            assert "message" in response2

    async def test_chat_with_system_message(self, ollama_server):
        """Test chat workflow with system message."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            messages = [
                {"role": "system", "content": "You are a math tutor"},
                {"role": "user", "content": "Explain addition"},
            ]

            response = await client.chat(messages)
            assert "message" in response


@pytest.mark.asyncio
class TestEndToEndModelManagement:
    """End-to-end tests for model management workflow."""

    async def test_list_models_workflow(self, ollama_server):
        """Test complete list models workflow."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            models = await client.list_models()

            assert isinstance(models, list)
            assert len(models) >= 2
            assert all("name" in m for m in models)

    async def test_get_model_info_workflow(self, ollama_server):
        """Test complete get model info workflow."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # First call - should fetch models
            model_info1 = await client.get_model_info("qwen2.5vl:7b")

            assert model_info1 is not None
            assert model_info1["name"] == "qwen2.5vl:7b"

            # Second call - should use cache
            model_info2 = await client.get_model_info("qwen2.5vl:7b")

            assert model_info1 == model_info2


@pytest.mark.asyncio
class TestEndToEndStreaming:
    """End-to-end tests for streaming workflows."""

    async def test_generate_stream_workflow(self, ollama_server):
        """Test complete streaming generation workflow."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            chunks = []
            async for chunk in client.generate_stream("Stream test"):
                chunks.append(chunk)
                if chunk.get("done"):
                    break

            assert len(chunks) > 0
            # Should have received chunks
            assert any("chunk" in c for c in chunks)

    async def test_chat_stream_workflow(self, ollama_server):
        """Test complete streaming chat workflow."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            messages = [{"role": "user", "content": "Stream chat test"}]
            chunks = []
            async for chunk in client.chat_stream(messages):
                chunks.append(chunk)
                if chunk.get("done"):
                    break

            assert len(chunks) > 0


@pytest.mark.asyncio
class TestEndToEndConcurrency:
    """End-to-end tests for concurrent operations."""

    async def test_concurrent_generate_and_chat(self, ollama_server):
        """Test concurrent generate and chat operations."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            generate_task = client.generate("Generate test")
            chat_task = client.chat([{"role": "user", "content": "Chat test"}])

            generate_result, chat_result = await asyncio.gather(generate_task, chat_task)

            assert isinstance(generate_result, GenerateResponse)
            assert isinstance(chat_result, dict)

    async def test_mixed_operations_workflow(self, ollama_server):
        """Test workflow with mixed operations."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # List models
            models = await client.list_models()
            assert len(models) > 0

            # Generate with first model
            response1 = await client.generate("Test 1", model=models[0]["name"])
            assert isinstance(response1, GenerateResponse)

            # Chat with same model
            response2 = await client.chat(
                [{"role": "user", "content": "Test 2"}], model=models[0]["name"]
            )
            assert isinstance(response2, dict)

            # Get model info
            model_info = await client.get_model_info(models[0]["name"])
            assert model_info is not None


@pytest.mark.asyncio
class TestEndToEndErrorRecovery:
    """End-to-end tests for error recovery workflows."""

    async def test_recovery_after_temporary_failure(self, ollama_server):
        """Test that client recovers after temporary failure."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # First request fails
            ollama_server.state["generate_failures"] = 1
            try:
                await client.generate("This will fail")
            except Exception:
                pass

            # Second request succeeds
            ollama_server.state["generate_failures"] = 0
            response = await client.generate("This should succeed")
            assert isinstance(response, GenerateResponse)

    async def test_health_check_workflow(self, ollama_server):
        """Test health check workflow."""
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # Healthy service
            health1 = await client.health_check()
            assert health1 is True

            # Unhealthy service
            ollama_server.state["tags_status"] = 500
            health2 = await client.health_check()
            assert health2 is False

            # Restore health
            ollama_server.state["tags_status"] = 200
            health3 = await client.health_check()
            assert health3 is True


class TestSyncClientIntegration:
    """Integration tests for synchronous client."""

    def test_sync_client_full_workflow(self, ollama_server):
        """Test complete workflow with synchronous client."""
        from shared_ollama import OllamaConfig

        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        # List models
        models = client.list_models()
        assert len(models) >= 2

        # Generate
        response = client.generate("Integration test")
        assert isinstance(response, GenerateResponse)
        assert response.text.startswith("ECHO: Integration test")

        # Chat
        chat_response = client.chat([{"role": "user", "content": "Chat test"}])
        assert isinstance(chat_response, dict)

        # Health check
        health = client.health_check()
        assert health is True

    def test_sync_client_with_options(self, ollama_server):
        """Test synchronous client with all options."""
        from shared_ollama import GenerateOptions, OllamaConfig

        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        options = GenerateOptions(
            temperature=0.7,
            top_p=0.95,
            max_tokens=50,
        )

        response = client.generate("Test with options", options=options)
        assert isinstance(response, GenerateResponse)


@pytest.mark.asyncio
class TestMetricsIntegration:
    """Integration tests for metrics collection across workflows."""

    async def test_metrics_across_multiple_operations(self, ollama_server):
        """Test that metrics are collected across multiple operations."""
        MetricsCollector.reset()
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            # Mix of operations
            await client.generate("Test 1")
            await client.chat([{"role": "user", "content": "Test 2"}])
            await client.list_models()
            await client.generate("Test 3")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 4
        assert metrics.requests_by_operation["generate"] >= 2
        assert metrics.requests_by_operation["chat"] >= 1
        assert metrics.requests_by_operation["list_models"] >= 1

    async def test_metrics_with_time_window(self, ollama_server):
        """Test that metrics filtering by time window works."""
        MetricsCollector.reset()
        config = AsyncOllamaConfig(base_url=ollama_server.base_url)
        async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
            await client.generate("Test")

        # Get metrics for last minute
        metrics = MetricsCollector.get_metrics(window_minutes=1)
        assert metrics.total_requests >= 1

        # Get metrics for last hour
        metrics_hour = MetricsCollector.get_metrics(window_minutes=60)
        assert metrics_hour.total_requests >= 1
