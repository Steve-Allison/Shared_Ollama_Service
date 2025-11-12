"""
Resilience patterns for the Shared Ollama Service clients.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

import requests

from shared_ollama.client.sync import GenerateResponse, OllamaConfig, SharedOllamaClient

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_timeout: float = 10.0


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_open_time: float | None = None

    def can_proceed(self) -> bool:
        match self.state:
            case CircuitState.CLOSED:
                return True
            case CircuitState.OPEN:
                if self.last_open_time and time.time() - self.last_open_time > self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker: moving to HALF_OPEN state")
                    return True
                return False
            case CircuitState.HALF_OPEN:
                return True
            case _:
                return False

    def record_success(self) -> None:
        match self.state:
            case CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker: moving to CLOSED state")
            case CircuitState.CLOSED:
                self.failure_count = 0
            case _:
                pass

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()

        match self.state:
            case CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.last_open_time = time.time()
                    logger.warning("Circuit breaker: moving to OPEN state")
            case CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.last_open_time = time.time()
                self.success_count = 0
                logger.warning("Circuit breaker: moving back to OPEN state")
            case _:
                pass


def exponential_backoff_retry(
    func: Callable[[], _T],
    config: RetryConfig | None = None,
    exceptions: tuple[type[Exception], ...] = (requests.RequestException,),
) -> _T:
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries):
        try:
            return func()
        except exceptions as exc:
            last_exception = exc

            if attempt < config.max_retries - 1:
                delay = min(config.initial_delay * (config.exponential_base**attempt), config.max_delay)
                if config.jitter:
                    delay *= 0.5 + random.random() * 0.5  # noqa: S311

                logger.warning(
                    "Retry attempt %s/%s after %.2fs: %s",
                    attempt + 1,
                    config.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.exception("All %s retry attempts failed", config.max_retries)

    if last_exception is None:
        msg = "All retries exhausted but no exception was captured"
        raise RuntimeError(msg)
    raise last_exception


class ResilientOllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        self.base_url = base_url
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        self.client = SharedOllamaClient(OllamaConfig(base_url=base_url), verify_on_init=False)

    def _execute_with_resilience(self, operation: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        if not self.circuit_breaker.can_proceed():
            msg = "Circuit breaker is OPEN - service is unavailable"
            raise ConnectionError(msg)

        try:
            result = exponential_backoff_retry(
                lambda: operation(*args, **kwargs),
                config=self.retry_config,
            )
            self.circuit_breaker.record_success()
        except (requests.RequestException, ConnectionError, TimeoutError):
            self.circuit_breaker.record_failure()
            raise
        else:
            return result

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> GenerateResponse:
        return self._execute_with_resilience(self.client.generate, prompt, model=model, **kwargs)

    def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        return self._execute_with_resilience(self.client.chat, messages, model=model, **kwargs)

    def health_check(self) -> bool:
        try:
            return self._execute_with_resilience(self.client.health_check)
        except (requests.RequestException, ConnectionError, TimeoutError) as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    def list_models(self) -> list[dict[str, Any]]:
        return self._execute_with_resilience(self.client.list_models)


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "ResilientOllamaClient",
    "RetryConfig",
    "exponential_backoff_retry",
]

