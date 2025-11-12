import pytest

import requests

from shared_ollama import (
    CircuitBreakerConfig,
    CircuitState,
    ResilientOllamaClient,
    RetryConfig,
)


def test_resilient_client_retries_until_success(ollama_server):
    ollama_server.state["generate_failures"] = 2

    client = ResilientOllamaClient(
        base_url=ollama_server.base_url,
        retry_config=RetryConfig(
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.02,
            exponential_base=2.0,
            jitter=False,
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=1,
            timeout=0.1,
        ),
    )

    response = client.generate("retry please")
    assert "retry please" in response.text
    assert ollama_server.state["generate_failures"] == 0


def test_circuit_breaker_opens_after_failures(ollama_server):
    ollama_server.state.update({"generate_failures": 5})

    client = ResilientOllamaClient(
        base_url=ollama_server.base_url,
        retry_config=RetryConfig(
            max_retries=2,
            initial_delay=0.0,
            max_delay=0.0,
            jitter=False,
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.2,
        ),
    )

    with pytest.raises(requests.RequestException):
        client.generate("trigger failures 1")

    with pytest.raises(requests.RequestException):
        client.generate("trigger failures 2")

    assert client.circuit_breaker.state == CircuitState.OPEN
    assert not client.circuit_breaker.can_proceed()

