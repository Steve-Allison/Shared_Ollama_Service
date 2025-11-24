"""Resilience helpers (retry + circuit breaker) for the Shared Ollama clients."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

import requests
from circuitbreaker import circuit
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared_ollama.client.sync import GenerateResponse, OllamaConfig, SharedOllamaClient

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
ExceptionTuple: TypeAlias = tuple[type[BaseException], ...]


@dataclass(slots=True, frozen=True)
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: ExceptionTuple = (ConnectionError,)


def exponential_backoff_retry(
    func: Callable[[], _T],
    *,
    config: RetryConfig | None = None,
    exceptions: ExceptionTuple | None = None,
) -> _T:
    """Invoke ``func`` with exponential-backoff retries (powered by tenacity)."""

    cfg = config or RetryConfig()
    retryable = exceptions or (ConnectionError, TimeoutError)
    retrying = Retrying(
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential(
            multiplier=cfg.initial_delay,
            min=cfg.initial_delay,
            max=cfg.max_delay,
        ),
        retry=retry_if_exception_type(retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    return retrying(func)


class ResilientOllamaClient:
    """SharedOllamaClient facade with baked-in retry + circuit-breaker policies."""

    __slots__ = (
        "base_url",
        "retry_config",
        "circuit_breaker_config",
        "client",
        "_retryable_exceptions",
        "_circuit_decorator",
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize resilient Ollama client.

        Creates a new resilient client with configurable retry and circuit
        breaker settings. The underlying SharedOllamaClient is created
        without initial verification (verify_on_init=False) for faster startup.

        Args:
            base_url: Base URL for Ollama service. Format: "http://host:port"
                or "https://host:port". Default: "http://localhost:11434".
            retry_config: Retry configuration for exponential backoff. None
                uses default RetryConfig() (3 retries, 1s initial delay).
            circuit_breaker_config: Circuit breaker configuration. None uses
                default CircuitBreakerConfig() (5 failures threshold, 60s recovery).
        """
        self.base_url = base_url
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.client = SharedOllamaClient(OllamaConfig(base_url=base_url), verify_on_init=False)
        self._retryable_exceptions: ExceptionTuple = (
            ConnectionError,
            TimeoutError,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
        )
        self._circuit_decorator = circuit(
            failure_threshold=self.circuit_breaker_config.failure_threshold,
            recovery_timeout=self.circuit_breaker_config.recovery_timeout,
            expected_exception=self.circuit_breaker_config.expected_exception,
        )

    def _execute_with_resilience(
        self,
        operation: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Apply circuit breaker + retry around ``operation``."""

        guarded = self._circuit_decorator(lambda: operation(*args, **kwargs))

        try:
            return exponential_backoff_retry(
                guarded,
                config=self.retry_config,
                exceptions=self._retryable_exceptions,
            )
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(f"Request failed: {exc!s}") from exc

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> GenerateResponse:
        """Generate text with resilience patterns.

        Args:
            prompt: Text prompt for generation.
            model: Model name. Uses default if None.
            **kwargs: Additional generation parameters passed to underlying client.

        Returns:
            GenerateResponse with generated text.

        Raises:
            ConnectionError: If circuit breaker is OPEN.
            ConnectionError: If all retries fail.
            TimeoutError: If operation times out.
        """
        return self._execute_with_resilience(self.client.generate, prompt, model=model, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat completion with resilience patterns.

        Args:
            messages: List of chat messages with 'role' and 'content' keys.
            model: Model name. Uses default if None.
            **kwargs: Additional chat parameters passed to underlying client.

        Returns:
            Dictionary with chat response.

        Raises:
            ConnectionError: If circuit breaker is OPEN.
            ConnectionError: If all retries fail.
            TimeoutError: If operation times out.
        """
        return self._execute_with_resilience(self.client.chat, messages, model=model, **kwargs)

    def health_check(self) -> bool:
        """Perform health check with resilience patterns.

        Returns:
            True if service is healthy, False otherwise. Never raises
            exceptions - returns False on any error.

        Side effects:
            Updates circuit breaker state based on health check result.
        """
        try:
            return self._execute_with_resilience(self.client.health_check)
        except (ConnectionError, TimeoutError) as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models with resilience patterns.

        Returns:
            List of model dictionaries.

        Raises:
            ConnectionError: If circuit breaker is OPEN.
            ConnectionError: If all retries fail.
            TimeoutError: If operation times out.
        """
        return self._execute_with_resilience(self.client.list_models)


__all__ = [
    "CircuitBreakerConfig",
    "ResilientOllamaClient",
    "RetryConfig",
    "exponential_backoff_retry",
]
