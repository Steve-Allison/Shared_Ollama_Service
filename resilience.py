"""
Resilience Features for Shared Ollama Service
==============================================

Provides enhanced resilience patterns for the Ollama client:
- Exponential backoff retry
- Circuit breaker
- Connection pooling
- Request timeout handling

Usage:
    from resilience import ResilientOllamaClient

    client = ResilientOllamaClient()
    response = client.generate("Hello!")
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import requests

from shared_ollama_client import OllamaConfig, SharedOllamaClient

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes
    timeout: float = 60.0  # Time before attempting half-open
    half_open_timeout: float = 10.0  # Time to test in half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    Prevents making requests when the service is known to be failing.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.last_open_time: float | None = None

    def can_proceed(self) -> bool:
        """
        Check if request can proceed.

        Returns:
            True if request should proceed, False if circuit is open
        """
        match self.state:
            case CircuitState.CLOSED:
                return True
            case CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_open_time and time.time() - self.last_open_time > self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker: Moving to HALF_OPEN state")
                    return True
                return False
            case CircuitState.HALF_OPEN:
                return True
            case _:
                return False

    def record_success(self) -> None:
        """Record a successful request."""
        match self.state:
            case CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker: Moving to CLOSED state")
            case CircuitState.CLOSED:
                self.failure_count = 0
            case _:
                pass

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        match self.state:
            case CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    self.last_open_time = time.time()
                    logger.warning("Circuit breaker: Moving to OPEN state")
            case CircuitState.HALF_OPEN:
                # Failure in half-open, back to open
                self.state = CircuitState.OPEN
                self.last_open_time = time.time()
                self.success_count = 0
                logger.warning("Circuit breaker: Moving back to OPEN state")
            case _:
                pass


def exponential_backoff_retry[T](
    func: Callable[[], T],
    config: RetryConfig | None = None,
    exceptions: tuple[type[Exception], ...] = (requests.RequestException,),
) -> T:
    """
    Execute function with exponential backoff retry.

    Args:
        func: Function to execute
        config: Retry configuration
        exceptions: Exception types to retry on

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e

            if attempt < config.max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(
                    config.initial_delay * (config.exponential_base**attempt),
                    config.max_delay,
                )

                # Add jitter if enabled
                if config.jitter:
                    delay *= 0.5 + random.random() * 0.5  # noqa: S311

                logger.warning(
                    f"Retry attempt {attempt + 1}/{config.max_retries} after {delay:.2f}s: {e}"
                )
                time.sleep(delay)
            else:
                logger.exception(f"All {config.max_retries} retry attempts failed")

    # All retries exhausted
    if last_exception is None:
        raise RuntimeError("All retries exhausted but no exception was captured")
    raise last_exception


class ResilientOllamaClient:
    """
    Enhanced Ollama client with resilience features.

    This wraps the standard SharedOllamaClient with:
    - Exponential backoff retry
    - Circuit breaker pattern
    - Better error handling

    Example:
        >>> from resilience import ResilientOllamaClient
        >>> client = ResilientOllamaClient()
        >>> response = client.generate("Hello!")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize resilient client.

        Args:
            base_url: Ollama service URL
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.base_url = base_url
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        self.client = SharedOllamaClient(OllamaConfig(base_url=base_url), verify_on_init=False)

    def _execute_with_resilience(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute operation with resilience features.

        Args:
            operation: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Operation result
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            msg = "Circuit breaker is OPEN - service is unavailable"
            raise ConnectionError(msg)

        # Execute with retry
        try:
            result = exponential_backoff_retry(
                lambda: operation(*args, **kwargs),
                config=self.retry_config,
            )
            self.circuit_breaker.record_success()
        except Exception:
            self.circuit_breaker.record_failure()
            raise
        else:
            return result

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> Any:
        """
        Generate text with resilience.

        Args:
            prompt: The prompt to generate from
            model: Model to use (optional)
            **kwargs: Additional arguments

        Returns:
            GenerateResponse
        """
        return self._execute_with_resilience(self.client.generate, prompt, model=model, **kwargs)

    def chat(self, messages: list[dict[str, str]], model: str | None = None, **kwargs: Any) -> Any:
        """
        Chat with model with resilience.

        Args:
            messages: List of message dicts
            model: Model to use (optional)
            **kwargs: Additional arguments

        Returns:
            Chat response
        """
        return self._execute_with_resilience(self.client.chat, messages, model=model, **kwargs)

    def health_check(self) -> bool:
        """
        Perform health check with resilience.

        Returns:
            True if service is healthy
        """
        try:
            return self._execute_with_resilience(self.client.health_check)
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """
        List models with resilience.

        Returns:
            List of model information
        """
        return self._execute_with_resilience(self.client.list_models)


if __name__ == "__main__":
    # Example usage
    print("Resilience Features Example")
    print("=" * 40)

    client = ResilientOllamaClient()

    # This will use exponential backoff and circuit breaker
    try:
        response = client.generate("Hello, world!")
        print(f"Response: {response.text}")
        print(f"Circuit breaker state: {client.circuit_breaker.state}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Circuit breaker state: {client.circuit_breaker.state}")
