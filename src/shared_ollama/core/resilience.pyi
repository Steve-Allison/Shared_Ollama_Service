"""Type stubs for shared_ollama.core.resilience module."""

from collections.abc import Callable
from typing import Any, TypeVar

from shared_ollama.client.sync import GenerateResponse

_T = TypeVar("_T")

class RetryConfig:
    max_retries: int
    initial_delay: float
    max_delay: float

class CircuitBreakerConfig:
    failure_threshold: int
    recovery_timeout: float
    expected_exception: type[Exception] | tuple[type[Exception], ...]

def exponential_backoff_retry(
    func: Callable[[], _T],
    config: RetryConfig | None = ...,
    exceptions: tuple[type[Exception], ...] | None = ...,
) -> _T: ...

class ResilientOllamaClient:
    def __init__(
        self,
        base_url: str = ...,
        retry_config: RetryConfig | None = ...,
        circuit_breaker_config: CircuitBreakerConfig | None = ...,
    ) -> None: ...
    def generate(
        self,
        prompt: str,
        model: str | None = ...,
        **kwargs: Any,
    ) -> GenerateResponse: ...
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = ...,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def health_check(self) -> bool: ...
    def list_models(self) -> list[dict[str, Any]]: ...
