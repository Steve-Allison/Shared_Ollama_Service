import httpx
import pytest

from shared_ollama import MetricsCollector
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_async_client_generate_and_chat(ollama_server):
    MetricsCollector.reset()
    config = AsyncOllamaConfig(base_url=ollama_server.base_url)

    async with AsyncSharedOllamaClient(config=config, verify_on_init=True) as client:
        models = await client.list_models()
        assert any(model["name"] == "qwen2.5vl:7b" for model in models)

        generate_response = await client.generate("Async ping?")
        assert "Async ping?" in generate_response.text
        assert generate_response.model == "qwen2.5vl:7b"

        chat_response = await client.chat([{"role": "user", "content": "Hello async"}])
        assert chat_response["message"]["content"].startswith("Echo:")

    metrics = MetricsCollector.get_metrics()
    assert metrics.total_requests >= 2
    assert metrics.successful_requests == metrics.total_requests


async def test_async_client_http_error_surface(ollama_server):
    ollama_server.state["generate_failures"] = 1
    config = AsyncOllamaConfig(base_url=ollama_server.base_url)

    async with AsyncSharedOllamaClient(config=config, verify_on_init=False) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("This should fail")

