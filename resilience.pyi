"""Type stubs for resilience module."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED: str
    OPEN: str
    HALF_OPEN: str

class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int
    initial_delay: float
    max_delay: float
    exponential_base: float
    jitter: bool

class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int
    success_threshold: int
    timeout: float
    half_open_timeout: float

class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""

    def __init__(
        self,
        config: CircuitBreakerConfig | None = ...,
    ) -> None: ...
    def can_proceed(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...

def exponential_backoff_retry(
    func: Callable[[], Any],
    config: RetryConfig | None = ...,
    exceptions: tuple[type[Exception], ...] = ...,
) -> Any: ...

class ResilientOllamaClient:
    """Enhanced Ollama client with resilience features."""

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
    ) -> Any: ...
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = ...,
        **kwargs: Any,
    ) -> Any: ...
    def health_check(self) -> bool: ...
    def list_models(self) -> list[dict[str, Any]]: ...
