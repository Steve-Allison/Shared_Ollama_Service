"""Resilience patterns for the Shared Ollama Service clients.

This module provides circuit breaker and retry patterns for building resilient
clients that can handle transient failures gracefully.

Key components:
    - CircuitBreaker: Prevents cascading failures by opening circuit after
      threshold failures
    - exponential_backoff_retry: Retries operations with exponential backoff
      and jitter
    - ResilientOllamaClient: Client wrapper combining both patterns

Key behaviors:
    - Circuit breaker has three states: CLOSED, OPEN, HALF_OPEN
    - Retry logic uses exponential backoff with configurable jitter
    - All operations are thread-safe for concurrent use
    - Automatic state transitions based on success/failure patterns
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
    """Circuit breaker state enumeration.

    States:
        CLOSED: Normal operation, requests allowed
        OPEN: Circuit open, requests blocked (too many failures)
        HALF_OPEN: Testing state, limited requests allowed to test recovery
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True, frozen=True)
class RetryConfig:
    """Configuration for exponential backoff retry logic.

    Immutable configuration object for retry behavior. All time values
    are in seconds.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
        initial_delay: Initial delay before first retry in seconds (default: 1.0).
        max_delay: Maximum delay cap in seconds (default: 60.0).
        exponential_base: Base multiplier for exponential backoff (default: 2.0).
        jitter: Whether to add random jitter to delays to prevent thundering
            herd (default: True).
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern.

    Immutable configuration object for circuit breaker behavior. All time
    values are in seconds.

    Attributes:
        failure_threshold: Number of consecutive failures before opening
            circuit (default: 5).
        success_threshold: Number of consecutive successes needed to close
            circuit from HALF_OPEN state (default: 2).
        timeout: Time in seconds before attempting HALF_OPEN state after
            circuit opens (default: 60.0).
        half_open_timeout: Timeout for HALF_OPEN state attempts (default: 10.0).
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    half_open_timeout: float = 10.0


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls.

    Implements the circuit breaker pattern to prevent cascading failures.
    Automatically transitions between states based on success/failure patterns.

    Attributes:
        config: Circuit breaker configuration.
        state: Current circuit breaker state (CircuitState enum).
        failure_count: Current number of consecutive failures.
        success_count: Current number of consecutive successes (used in
            HALF_OPEN state).
        last_failure_time: Timestamp of last failure (monotonic time).
        last_open_time: Timestamp when circuit was opened (monotonic time).

    Thread safety:
        Not thread-safe. Use from a single thread or protect with locks
        if accessing from multiple threads.

    State machine:
        CLOSED -> OPEN: After failure_threshold failures
        OPEN -> HALF_OPEN: After timeout seconds
        HALF_OPEN -> CLOSED: After success_threshold successes
        HALF_OPEN -> OPEN: On any failure
    """

    __slots__ = (
        "config",
        "state",
        "failure_count",
        "success_count",
        "last_failure_time",
        "last_open_time",
    )

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration. If None, uses default
                CircuitBreakerConfig().
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_open_time: float | None = None

    def can_proceed(self) -> bool:
        """Check if request can proceed based on circuit state.

        Automatically transitions from OPEN to HALF_OPEN if timeout has
        elapsed. Uses pattern matching for clean state-based logic.

        Returns:
            True if request can proceed, False if circuit is OPEN and
            timeout has not elapsed.

        Side effects:
            May transition state from OPEN to HALF_OPEN if timeout elapsed.
        """
        match self.state:
            case CircuitState.CLOSED:
                return True
            case CircuitState.OPEN:
                if (
                    self.last_open_time is not None
                    and time.monotonic() - self.last_open_time > self.config.timeout
                ):
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
        """Record a successful operation.

        Updates circuit state based on current state and success threshold.
        Resets failure count in CLOSED state. Transitions HALF_OPEN to CLOSED
        after success_threshold consecutive successes.

        Side effects:
            - Updates success_count and failure_count
            - May transition state from HALF_OPEN to CLOSED
        """
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
        """Record a failed operation.

        Updates circuit state based on failure threshold. Transitions CLOSED
        to OPEN after failure_threshold consecutive failures. Transitions
        HALF_OPEN to OPEN immediately on any failure.

        Side effects:
            - Increments failure_count
            - Updates last_failure_time and last_open_time
            - May transition state from CLOSED to OPEN or HALF_OPEN to OPEN
        """
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        match self.state:
            case CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.last_open_time = time.monotonic()
                    logger.warning("Circuit breaker: moving to OPEN state")
            case CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.last_open_time = time.monotonic()
                self.success_count = 0
                logger.warning("Circuit breaker: moving back to OPEN state")
            case _:
                pass


def exponential_backoff_retry(
    func: Callable[[], _T],
    config: RetryConfig | None = None,
    exceptions: tuple[type[Exception], ...] | None = None,
) -> _T:
    """Execute function with exponential backoff retry logic.

    Retries the function call with exponentially increasing delays between
    attempts. Adds random jitter to prevent thundering herd problems.

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
        # Default to generic exceptions instead of framework-specific ones
        exceptions = (ConnectionError, TimeoutError)
    last_exception: Exception | None = None

    for attempt in range(config.max_retries):
        try:
            return func()
        except exceptions as exc:
            last_exception = exc

            if attempt < config.max_retries - 1:
                delay = min(
                    config.initial_delay * (config.exponential_base**attempt),
                    config.max_delay,
                )
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
    """Ollama client with built-in resilience patterns.

    Combines circuit breaker and retry logic for robust service calls.
    Wraps SharedOllamaClient with automatic failure handling.

    Attributes:
        base_url: Base URL for Ollama service.
        retry_config: Retry configuration.
        circuit_breaker: Circuit breaker instance.
        client: Underlying SharedOllamaClient instance.

    Thread safety:
        Not thread-safe. Use from a single thread or protect with locks.
    """

    __slots__ = ("base_url", "retry_config", "circuit_breaker", "client")

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
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
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
            TimeoutError: If operation times out.

        Side effects:
            - Updates circuit breaker state based on success/failure
            - May sleep during retry attempts
        """
        if not self.circuit_breaker.can_proceed():
            msg = "Circuit breaker is OPEN - service is unavailable"
            raise ConnectionError(msg)

        try:
            # Include HTTPError in retry exceptions for transient server errors
            result = exponential_backoff_retry(
                lambda: operation(*args, **kwargs),
                config=self.retry_config,
                exceptions=(
                    ConnectionError,
                    TimeoutError,
                    requests.exceptions.HTTPError,  # Retry 5xx errors
                    requests.exceptions.RequestException,  # Retry network errors
                ),
            )
            self.circuit_breaker.record_success()
            return result
        except requests.exceptions.HTTPError as exc:
            # HTTP errors that weren't retried (e.g., 4xx client errors)
            self.circuit_breaker.record_failure()
            raise
        except requests.exceptions.RequestException as exc:
            # Network-level errors - convert to ConnectionError for consistency
            self.circuit_breaker.record_failure()
            raise ConnectionError(f"Request failed: {exc!s}") from exc
        except (ConnectionError, TimeoutError) as exc:
            self.circuit_breaker.record_failure()
            raise

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
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "ResilientOllamaClient",
    "RetryConfig",
    "exponential_backoff_retry",
]
