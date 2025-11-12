"""Type stubs for shared_ollama.client.async_client module."""

from typing import Any

from shared_ollama.client.sync import GenerateOptions, GenerateResponse

class AsyncOllamaConfig:
    base_url: str
    default_model: str
    timeout: int
    health_check_timeout: int
    verbose: bool
    max_retries: int
    retry_delay: float
    max_connections: int
    max_keepalive_connections: int
    max_concurrent_requests: int | None

class AsyncSharedOllamaClient:
    def __init__(
        self,
        config: AsyncOllamaConfig | None = ...,
        verify_on_init: bool = ...,
    ) -> None: ...
    async def __aenter__(self) -> AsyncSharedOllamaClient: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...
    async def close(self) -> None: ...
    async def list_models(self) -> list[dict[str, Any]]: ...
    async def generate(
        self,
        prompt: str,
        model: str | None = ...,
        system: str | None = ...,
        options: GenerateOptions | None = ...,
        stream: bool = ...,
    ) -> GenerateResponse: ...
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = ...,
        stream: bool = ...,
    ) -> dict[str, Any]: ...
    async def health_check(self) -> bool: ...
    async def get_model_info(self, model: str) -> dict[str, Any] | None: ...

