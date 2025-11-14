"""
Comprehensive behavioral tests for resilience patterns.

Tests focus on real retry behavior, circuit breaker state transitions,
timeout scenarios, and edge cases. Uses real HTTP requests with test server.
"""

import time

import pytest
import requests

from shared_ollama import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ResilientOllamaClient,
    RetryConfig,
    exponential_backoff_retry,
)


class TestExponentialBackoffRetry:
    """Behavioral tests for exponential_backoff_retry()."""

    def test_retry_succeeds_on_first_attempt(self):
        """Test that retry succeeds immediately when function succeeds."""
        call_count = [0]

        def successful_func():
            call_count[0] += 1
            return "success"

        result = exponential_backoff_retry(successful_func)
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_retries_on_failure_then_succeeds(self):
        """Test that retry retries on failure and eventually succeeds."""
        call_count = [0]

        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise requests.RequestException("Temporary failure")
            return "success"

        result = exponential_backoff_retry(
            failing_then_success,
            config=RetryConfig(max_retries=5, initial_delay=0.01, jitter=False),
        )
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhausts_after_max_retries(self):
        """Test that retry raises exception after max retries exhausted."""

        def always_fails():
            raise requests.RequestException("Permanent failure")

        with pytest.raises(requests.RequestException):
            exponential_backoff_retry(
                always_fails, config=RetryConfig(max_retries=3, initial_delay=0.01)
            )

    def test_retry_respects_exponential_backoff(self):
        """Test that retry uses exponential backoff with correct delays."""
        call_times = []

        def failing_func():
            call_times.append(time.perf_counter())
            if len(call_times) < 3:
                raise requests.RequestException("Temporary failure")
            return "success"

        start_time = time.perf_counter()
        exponential_backoff_retry(
            failing_func,
            config=RetryConfig(
                max_retries=5, initial_delay=0.1, exponential_base=2.0, jitter=False
            ),
        )

        # Check delays are exponential: 0.1s, 0.2s
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Allow some tolerance for timing
            assert 0.08 < delay1 < 0.15
            assert 0.18 < delay2 < 0.25

    def test_retry_respects_max_delay(self):
        """Test that retry respects max_delay cap."""
        call_times = []

        def failing_func():
            call_times.append(time.perf_counter())
            if len(call_times) < 3:
                raise requests.RequestException("Temporary failure")
            return "success"

        start_time = time.perf_counter()
        exponential_backoff_retry(
            failing_func,
            config=RetryConfig(
                max_retries=5,
                initial_delay=0.1,
                max_delay=0.15,
                exponential_base=2.0,
                jitter=False,
            ),
        )

        # Delays should be capped at max_delay
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay1 <= 0.16  # Allow tolerance
            assert delay2 <= 0.16  # Should be capped

    def test_retry_only_retries_specified_exceptions(self):
        """Test that retry only retries specified exception types."""

        def raises_value_error():
            raise ValueError("Not a RequestException")

        with pytest.raises(ValueError):
            exponential_backoff_retry(
                raises_value_error,
                config=RetryConfig(max_retries=3, initial_delay=0.01),
                exceptions=(requests.RequestException,),
            )


class TestCircuitBreaker:
    """Behavioral tests for CircuitBreaker."""

    def test_circuit_breaker_starts_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_proceed() is True
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_after_threshold_failures(self):
        """Test that circuit breaker opens after failure_threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)

        # Record failures up to threshold
        for i in range(3):
            cb.record_failure()
            if i < 2:
                assert cb.state == CircuitState.CLOSED
            else:
                assert cb.state == CircuitState.OPEN

        assert cb.state == CircuitState.OPEN
        assert cb.can_proceed() is False

    def test_circuit_breaker_resets_on_success_in_closed(self):
        """Test that circuit breaker resets failure count on success in CLOSED state."""
        cb = CircuitBreaker()

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_transitions_to_half_open_after_timeout(self):
        """Test that circuit breaker transitions to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        cb = CircuitBreaker(config=config)

        # Force to OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to HALF_OPEN
        assert cb.can_proceed() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_closes_after_success_threshold_in_half_open(self):
        """Test that circuit breaker closes after success_threshold in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=2, success_threshold=2, timeout=0.1
        )
        cb = CircuitBreaker(config=config)

        # Force to OPEN, then HALF_OPEN
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_proceed()  # Triggers transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Not enough yet

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_circuit_breaker_opens_immediately_on_failure_in_half_open(self):
        """Test that circuit breaker opens immediately on failure in HALF_OPEN."""
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        cb = CircuitBreaker(config=config)

        # Force to OPEN, then HALF_OPEN
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.can_proceed()  # Triggers transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

        # Single failure should open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.success_count == 0

    def test_circuit_breaker_does_not_open_before_threshold(self):
        """Test that circuit breaker stays closed before reaching threshold."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config=config)

        for i in range(4):
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count == i + 1

    def test_circuit_breaker_tracks_last_failure_time(self):
        """Test that circuit breaker tracks last failure timestamp."""
        cb = CircuitBreaker()

        assert cb.last_failure_time is None

        cb.record_failure()
        assert cb.last_failure_time is not None
        assert isinstance(cb.last_failure_time, float)

    def test_circuit_breaker_tracks_last_open_time(self):
        """Test that circuit breaker tracks when circuit was opened."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)

        assert cb.last_open_time is None

        cb.record_failure()
        cb.record_failure()  # Opens circuit

        assert cb.last_open_time is not None
        assert isinstance(cb.last_open_time, float)


class TestResilientOllamaClient:
    """Behavioral tests for ResilientOllamaClient integration."""

    def test_resilient_client_retries_until_success(self, ollama_server):
        """Test that resilient client retries until success."""
        ollama_server.state["generate_failures"] = 2

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(
                max_retries=5,
                initial_delay=0.01,
                max_delay=0.02,
                exponential_base=2.0,
                jitter=False,
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=10,  # High threshold to avoid opening
                success_threshold=1,
                timeout=0.1,
            ),
        )

        response = client.generate("retry please")
        assert "retry please" in response.text
        assert ollama_server.state["generate_failures"] == 0

    def test_resilient_client_opens_circuit_after_failures(self, ollama_server):
        """Test that resilient client opens circuit after threshold failures."""
        ollama_server.state["generate_failures"] = 10  # More than retries

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(
                max_retries=2, initial_delay=0.0, max_delay=0.0, jitter=False
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, success_threshold=1, timeout=0.2
            ),
        )

        # First failure
        with pytest.raises((requests.RequestException, ConnectionError)):
            client.generate("trigger failure 1")

        # Second failure should open circuit
        with pytest.raises((requests.RequestException, ConnectionError)):
            client.generate("trigger failure 2")

        assert client.circuit_breaker.state == CircuitState.OPEN
        assert not client.circuit_breaker.can_proceed()

    def test_resilient_client_blocks_when_circuit_open(self, ollama_server):
        """Test that resilient client blocks requests when circuit is open."""
        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, success_threshold=1, timeout=1.0
            ),
        )

        # Force circuit open
        client.circuit_breaker.state = CircuitState.OPEN
        client.circuit_breaker.last_open_time = time.monotonic()

        with pytest.raises(ConnectionError, match="Circuit breaker is OPEN"):
            client.generate("should be blocked")

    def test_resilient_client_records_success(self, ollama_server):
        """Test that resilient client records success and closes circuit."""
        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5, success_threshold=1, timeout=0.1
            ),
        )

        response = client.generate("success test")
        assert "success test" in response.text
        assert client.circuit_breaker.state == CircuitState.CLOSED
        assert client.circuit_breaker.failure_count == 0

    def test_resilient_client_handles_chat_requests(self, ollama_server):
        """Test that resilient client handles chat requests with retry."""
        ollama_server.state["chat_failures"] = 1

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=3, initial_delay=0.01),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5, success_threshold=1, timeout=0.1
            ),
        )

        messages = [{"role": "user", "content": "Chat test"}]
        response = client.chat(messages)
        assert "message" in response

    def test_resilient_client_health_check_never_raises(self, ollama_server):
        """Test that resilient client health_check() never raises exceptions."""
        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, success_threshold=1, timeout=0.1
            ),
        )

        # Even with circuit open, health_check should return False, not raise
        client.circuit_breaker.state = CircuitState.OPEN
        result = client.health_check()
        assert isinstance(result, bool)
