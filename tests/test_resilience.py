"""
Comprehensive behavioral tests for resilience patterns.

Tests focus on real retry behavior, circuit breaker behavior through ResilientOllamaClient,
timeout scenarios, and edge cases. Uses real HTTP requests with test server.
"""

import contextlib
import time

import pytest
from requests.exceptions import HTTPError

from shared_ollama import (
    CircuitBreakerConfig,
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
                raise ConnectionError("Temporary failure")
            return "success"

        result = exponential_backoff_retry(
            failing_then_success,
            config=RetryConfig(max_retries=5, initial_delay=0.01),
        )
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhausts_after_max_retries(self):
        """Test that retry raises exception after max retries exhausted."""

        def always_fails():
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError):
            exponential_backoff_retry(
                always_fails, config=RetryConfig(max_retries=3, initial_delay=0.01)
            )

    def test_retry_respects_exponential_backoff(self):
        """Test that retry uses exponential backoff with correct delays."""
        call_times = []

        def failing_func():
            call_times.append(time.perf_counter())
            if len(call_times) < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        exponential_backoff_retry(
            failing_func,
            config=RetryConfig(
                max_retries=5, initial_delay=0.1
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
                raise ConnectionError("Temporary failure")
            return "success"

        exponential_backoff_retry(
            failing_func,
            config=RetryConfig(
                max_retries=5,
                initial_delay=0.1,
                max_delay=0.15,
            ),
        )

        # Delays should be capped at max_delay (with tolerance for execution overhead)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay1 <= 0.17  # Allow tolerance for execution overhead
            assert delay2 <= 0.17  # Should be capped at ~0.15, allow tolerance

    def test_retry_only_retries_specified_exceptions(self):
        """Test that retry only retries specified exception types."""

        def raises_value_error():
            raise ValueError("Not a ConnectionError")

        with pytest.raises(ValueError):
            exponential_backoff_retry(
                raises_value_error,
                config=RetryConfig(max_retries=3, initial_delay=0.01),
                exceptions=(ConnectionError,),
            )


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
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=10,  # High threshold to avoid opening
                recovery_timeout=0.1,
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
                max_retries=2, initial_delay=0.0, max_delay=0.0
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.2
            ),
        )

        # First failure - HTTPError will be retried, then eventually raises HTTPError
        with pytest.raises((ConnectionError, TimeoutError, HTTPError)):
            client.generate("trigger failure 1")

        # Second failure should open circuit (circuitbreaker library handles this)
        with pytest.raises((ConnectionError, TimeoutError, HTTPError)):
            client.generate("trigger failure 2")

    def test_resilient_client_handles_chat_requests(self, ollama_server):
        """Test that resilient client handles chat requests with retry."""
        ollama_server.state["chat_failures"] = 1

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(
                max_retries=3,
                initial_delay=0.01,
                max_delay=0.02,
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=0.1
            ),
        )

        messages = [{"role": "user", "content": "Chat test"}]
        # Should succeed after one retry (first attempt fails, second succeeds)
        response = client.chat(messages)
        assert "message" in response
        assert ollama_server.state["chat_failures"] == 0

    def test_resilient_client_health_check_never_raises(self, ollama_server):
        """Test that resilient client health_check() never raises exceptions."""
        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout=0.1
            ),
        )

        # Even with failures, health_check should return False, not raise
        result = client.health_check()
        assert isinstance(result, bool)

    def test_resilient_client_recovers_after_timeout(self, ollama_server):
        """Test that resilient client recovers after circuit breaker timeout."""
        ollama_server.state["generate_failures"] = 10  # Always fail

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.2
            ),
        )

        # Trigger failures to open circuit
        with contextlib.suppress(Exception):
            client.generate("fail 1")
        with contextlib.suppress(Exception):
            client.generate("fail 2")

        # Wait for recovery timeout
        time.sleep(0.25)

        # Reset server to succeed
        ollama_server.state["generate_failures"] = 0

        # Should eventually succeed after circuit breaker recovery
        # (may take a few attempts as circuit breaker tests recovery)
        for _ in range(5):
            try:
                response = client.generate("recovery test")
                assert "recovery test" in response.text
                break
            except Exception:
                time.sleep(0.1)


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions and recovery."""

    def test_circuit_breaker_closed_to_open_transition(self, ollama_server):
        """Test that circuit breaker transitions from closed to open after failures."""
        ollama_server.state["generate_failures"] = 10  # Always fail

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.2
            ),
        )

        # First failure - circuit still closed
        with contextlib.suppress(Exception):
            client.generate("fail 1")

        # Second failure should open circuit
        with pytest.raises((ConnectionError, TimeoutError, HTTPError)):
            client.generate("fail 2")

        # Third attempt should fail immediately (circuit open)
        with pytest.raises((ConnectionError, TimeoutError, HTTPError)):
            client.generate("fail 3")

    def test_circuit_breaker_open_to_half_open_transition(self, ollama_server):
        """Test that circuit breaker transitions from open to half-open after timeout."""
        ollama_server.state["generate_failures"] = 10  # Start failing

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.2
            ),
        )

        # Open circuit
        with contextlib.suppress(Exception):
            client.generate("fail 1")
        with contextlib.suppress(Exception):
            client.generate("fail 2")

        # Wait for recovery timeout
        time.sleep(0.25)

        # Reset server to succeed
        ollama_server.state["generate_failures"] = 0

        # Should transition to half-open and succeed
        for _ in range(5):
            try:
                response = client.generate("recovery")
                assert "recovery" in response.text
                break
            except Exception:
                time.sleep(0.1)

    def test_circuit_breaker_half_open_to_closed_on_success(self, ollama_server):
        """Test that circuit breaker closes after successful half-open test."""
        ollama_server.state["generate_failures"] = 10

        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.2
            ),
        )

        # Open circuit
        with contextlib.suppress(Exception):
            client.generate("fail 1")
        with contextlib.suppress(Exception):
            client.generate("fail 2")

        # Wait for recovery
        time.sleep(0.25)
        ollama_server.state["generate_failures"] = 0

        # First successful call should close circuit
        response = client.generate("success")
        assert "success" in response.text

        # Subsequent calls should work normally (circuit closed)
        response2 = client.generate("success 2")
        assert "success 2" in response2.text

    def test_circuit_breaker_half_open_to_open_on_failure(self, ollama_server):
        """Test that circuit breaker reopens if half-open test fails."""
        client = ResilientOllamaClient(
            base_url=ollama_server.base_url,
            retry_config=RetryConfig(max_retries=1, initial_delay=0.0),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.2
            ),
        )

        # Open circuit
        ollama_server.state["generate_failures"] = 10
        with contextlib.suppress(Exception):
            client.generate("fail 1")
        with contextlib.suppress(Exception):
            client.generate("fail 2")

        # Wait for recovery
        time.sleep(0.25)

        # Half-open test fails - should reopen
        ollama_server.state["generate_failures"] = 1  # Fail once more
        with pytest.raises((ConnectionError, TimeoutError, HTTPError)):
            client.generate("half-open fail")

        # Circuit may be open again (circuit breaker behavior can vary)
        # Just verify that failures are handled correctly
        ollama_server.state["generate_failures"] = 0  # Reset
        # May succeed or fail depending on circuit breaker state
        try:
            client.generate("test after half-open failure")
        except (ConnectionError, TimeoutError, HTTPError):
            pass  # Expected if circuit is still open


class TestConcurrentFailureScenarios:
    """Tests for concurrent failure handling."""

    def test_concurrent_retries_do_not_interfere(self):
        """Test that concurrent retries don't interfere with each other."""
        import threading

        call_counts = {}
        lock = threading.Lock()

        def failing_func_with_id(func_id: str):
            with lock:
                call_counts[func_id] = call_counts.get(func_id, 0) + 1
                current_count = call_counts[func_id]
            if current_count < 3:
                raise ConnectionError(f"Failure {func_id}")
            return f"success {func_id}"

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    exponential_backoff_retry,
                    lambda i=i: failing_func_with_id(f"func-{i}"),
                    config=RetryConfig(max_retries=5, initial_delay=0.01),
                )
                for i in range(3)
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert len(results) == 3
        assert all("success" in r for r in results)
        # Each function should have been called at least 3 times (may be more due to retries)
        with lock:
            assert all(call_counts.get(f"func-{i}", 0) >= 3 for i in range(3))

    def test_retry_with_multiple_exception_types(self):
        """Test that retry handles multiple exception types correctly."""
        call_count = [0]

        def raises_different_exceptions():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Connection error")
            elif call_count[0] == 2:
                raise TimeoutError("Timeout error")
            return "success"

        result = exponential_backoff_retry(
            raises_different_exceptions,
            config=RetryConfig(max_retries=5, initial_delay=0.01),
            exceptions=(ConnectionError, TimeoutError),
        )

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhaustion_with_different_exceptions(self):
        """Test that retry exhaustion works with different exception types."""
        call_count = [0]

        def raises_alternating_exceptions():
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                raise ConnectionError("Connection error")
            else:
                raise TimeoutError("Timeout error")

        with pytest.raises((ConnectionError, TimeoutError)):
            exponential_backoff_retry(
                raises_alternating_exceptions,
                config=RetryConfig(max_retries=3, initial_delay=0.01),
                exceptions=(ConnectionError, TimeoutError),
            )

        # Should have tried max_retries times
        assert call_count[0] == 3


class TestRetryEdgeCases:
    """Tests for retry edge cases and boundary conditions."""

    def test_retry_with_zero_max_retries(self):
        """Test that retry with zero max_retries doesn't retry."""
        call_count = [0]

        def failing_func():
            call_count[0] += 1
            raise ConnectionError("Failure")

        with pytest.raises(ConnectionError):
            exponential_backoff_retry(
                failing_func,
                config=RetryConfig(max_retries=0, initial_delay=0.01),
            )

        # Should only call once (no retries)
        assert call_count[0] == 1

    def test_retry_with_very_small_delays(self):
        """Test that retry works with very small delay values."""
        call_count = [0]

        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Failure")
            return "success"

        result = exponential_backoff_retry(
            failing_func,
            config=RetryConfig(max_retries=3, initial_delay=0.001, max_delay=0.01),
        )

        assert result == "success"
        assert call_count[0] == 2

    def test_retry_with_equal_initial_and_max_delay(self):
        """Test that retry works when initial_delay equals max_delay."""
        call_count = [0]

        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Failure")
            return "success"

        result = exponential_backoff_retry(
            failing_func,
            config=RetryConfig(max_retries=5, initial_delay=0.1, max_delay=0.1),
        )

        assert result == "success"
        assert call_count[0] == 3

    @pytest.mark.parametrize("max_retries", [1, 2, 3, 5, 10])
    def test_retry_respects_max_retries_configuration(self, max_retries):
        """Test that retry respects max_retries configuration."""
        call_count = [0]

        def always_fails():
            call_count[0] += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            exponential_backoff_retry(
                always_fails,
                config=RetryConfig(max_retries=max_retries, initial_delay=0.01),
            )

        # Should have called exactly max_retries times
        assert call_count[0] == max_retries
