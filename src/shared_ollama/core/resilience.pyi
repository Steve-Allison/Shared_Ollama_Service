"""Type stubs for shared_ollama.core.resilience module."""

from typing import Any, Callable, TypeVar

from shared_ollama.client.sync import GenerateResponse

_T = TypeVar("_T")

class CircuitState:
    CLOSED: str
    OPEN: str
    HALF_OPEN: str

class RetryConfig:
    max_retries: int
    initial_delay: float
    max_delay: float
    exponential_base: float
    jitter: bool

class CircuitBreakerConfig:
    failure_threshold: int
    success_threshold: int
    timeout: float
    half_open_timeout: float

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = ...) -> None: ...
    def can_proceed(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...

def exponential_backoff_retry(
    func: Callable[[], _T],
    config: RetryConfig | None = ...,
    exceptions: tuple[type[Exception], ...] = ...,
) -> _T: ...

class ResilientOllamaClient:
    def __init__(
        self,
        base_url: str = ...,
        retry_config: RetryConfig | None = ...,
        circuit_breaker_config: CircuitBreakerConfig | None = ...,
    ) -> None: ...
    def generate(self, prompt: str, model: str | None = ..., **kwargs: Any) -> GenerateResponse: ...
    def chat(self, messages: list[dict[str, str]], model: str | None = ..., **kwargs: Any) -> dict[str, Any]: ...
    def health_check(self) -> bool: ...
    def list_models(self) -> list[dict[str, Any]]: ...

