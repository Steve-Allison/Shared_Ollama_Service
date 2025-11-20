"""Resilience patterns for the Shared Ollama Service clients.

This module provides circuit breaker and retry patterns for building resilient
clients that can handle transient failures gracefully.

Key components:
    - Circuit breaker: Prevents cascading failures by opening circuit after
      threshold failures (using circuitbreaker library)
    - exponential_backoff_retry: Retries operations with exponential backoff
      (using tenacity library, which handles jitter automatically)
    - ResilientOllamaClient: Client wrapper combining both patterns

Key behaviors:
    - Circuit breaker has three states: CLOSED, OPEN, HALF_OPEN
    - Retry logic uses exponential backoff (tenacity handles jitter automatically)
    - All operations are thread-safe for concurrent use
    - Automatic state transitions based on success/failure patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import requests
from circuitbreaker import circuit
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared_ollama.client.sync import OllamaConfig, SharedOllamaClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from shared_ollama.client.sync import GenerateResponse
else:  # pragma: no cover - for postponed annotations
    Callable = Any  # type: ignore[assignment]
    GenerateResponse = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass(slots=True, frozen=True)
class RetryConfig:
    """Configuration for exponential backoff retry logic.

    Immutable configuration object for retry behavior. All time values
    are in seconds. Uses tenacity library which handles exponential backoff
    and jitter automatically.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
        initial_delay: Initial delay before first retry in seconds (default: 1.0).
        max_delay: Maximum delay cap in seconds (default: 60.0).
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern.

    Immutable configuration object for circuit breaker behavior. All time
    values are in seconds.

    Attributes:
        failure_threshold: Number of consecutive failures before opening
            circuit (default: 5).
        recovery_timeout: Time in seconds before attempting HALF_OPEN state after
            circuit opens (default: 60.0).
        expected_exception: Exception type(s) that trigger circuit breaker
            (default: ConnectionError).
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type[Exception] | tuple[type[Exception], ...] = ConnectionError


def exponential_backoff_retry(
    func: Callable[[], _T],
    config: RetryConfig | None = None,
    exceptions: tuple[type[Exception], ...] | None = None,
) -> _T:
    """Execute function with exponential backoff retry logic.

    Uses tenacity library for robust retry handling with exponential backoff.
    Tenacity automatically handles jitter to prevent thundering herd problems.
    Retries the function call with exponentially increasing delays between attempts.

    Args:
        func: Function to execute. Must be callable with no arguments.
        config: Retry configuration. If None, uses default RetryConfig().
        exceptions: Tuple of exception types to catch and retry. Defaults to
            (ConnectionError, TimeoutError) if None.

    Returns:
        Result of function execution.

    Raises:
        Last exception encountered if all retries are exhausted. The exception
        type will be one of the types in the exceptions tuple.

    Side effects:
        - Sleeps between retry attempts
        - Logs warning messages for each retry attempt
        - Logs exception on final failure

    Example:
        >>> def fetch_data():
        ...     # Example function that may raise ConnectionError or TimeoutError
        ...     return {"data": "example"}
        >>> result = exponential_backoff_retry(fetch_data)
    """
    config = config or RetryConfig()
    if exceptions is None:
        exceptions = (ConnectionError, TimeoutError)

    # Create retry decorator with configuration
    retry_decorator = retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(
            multiplier=config.initial_delay,
            min=config.initial_delay,
            max=config.max_delay,
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    # Apply decorator and execute
    retry_func = retry_decorator(func)
    return retry_func()


class ResilientOllamaClient:
    """Ollama client with built-in resilience patterns.

    Combines circuit breaker and retry logic for robust service calls.
    Wraps SharedOllamaClient with automatic failure handling.

    Attributes:
        base_url: Base URL for Ollama service.
        retry_config: Retry configuration.
        circuit_breaker_config: Circuit breaker configuration.
        client: Underlying SharedOllamaClient instance.

    Thread safety:
        Thread-safe. Uses circuitbreaker library which is thread-safe.
    """

    __slots__ = ("base_url", "circuit_breaker_config", "client", "retry_config")

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize resilient Ollama client.

        Args:
            base_url: Base URL for Ollama service.
            retry_config: Retry configuration. If None, uses default RetryConfig().
            circuit_breaker_config: Circuit breaker configuration. If None,
                uses default CircuitBreakerConfig().
        """
        self.base_url = base_url
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.client = SharedOllamaClient(
            OllamaConfig(base_url=base_url), verify_on_init=False
        )

    def _execute_with_resilience(
        self, operation: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T:
        """Execute operation with circuit breaker and retry logic.

        Args:
            operation: Function to execute.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Result of operation.

        Raises:
            ConnectionError: If circuit breaker is OPEN.
            ConnectionError: If all retries fail.
            TimeoutError: If operation times out.

        Side effects:
            - Updates circuit breaker state based on success/failure
            - May sleep during retry attempts
        """
        # Create circuit breaker decorator
        cb_decorator = circuit(
            failure_threshold=self.circuit_breaker_config.failure_threshold,
            recovery_timeout=self.circuit_breaker_config.recovery_timeout,
            expected_exception=self.circuit_breaker_config.expected_exception,
        )

        # Wrap operation with circuit breaker
        @cb_decorator
        def wrapped_operation() -> _T:
            return operation(*args, **kwargs)

        # Apply retry logic to wrapped operation
        try:
            return exponential_backoff_retry(
                wrapped_operation,
                config=self.retry_config,
                exceptions=(
                    ConnectionError,
                    TimeoutError,
                    requests.exceptions.HTTPError,
                    requests.exceptions.RequestException,
                ),
            )
        except requests.exceptions.HTTPError:
            # HTTP errors that weren't retried (e.g., 4xx client errors)
            raise
        except requests.exceptions.RequestException as exc:
            # Network-level errors - convert to ConnectionError for consistency
            raise ConnectionError(f"Request failed: {exc!s}") from exc

    def generate(
        self, prompt: str, model: str | None = None, **kwargs: Any
    ) -> GenerateResponse:
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
        return self._execute_with_resilience(
            self.client.generate, prompt, model=model, **kwargs
        )

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
        return self._execute_with_resilience(
            self.client.chat, messages, model=model, **kwargs
        )

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
