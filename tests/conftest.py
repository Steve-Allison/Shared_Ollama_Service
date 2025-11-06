"""
Pytest configuration and fixtures for Shared Ollama Service tests.
"""

from unittest.mock import Mock, patch

import pytest

from shared_ollama_client import OllamaConfig, SharedOllamaClient


@pytest.fixture
def mock_ollama_url():
    """Mock Ollama service URL."""
    return "http://localhost:11434"


@pytest.fixture
def ollama_config(mock_ollama_url):
    """Create OllamaConfig for testing."""
    return OllamaConfig(base_url=mock_ollama_url)


@pytest.fixture
def mock_client(ollama_config):
    """Create a SharedOllamaClient with mocked connection verification."""
    with patch("shared_ollama_client.SharedOllamaClient._verify_connection"):
        client = SharedOllamaClient(config=ollama_config, verify_on_init=False)
        # Ensure session is a Mock for testing
        client.session = Mock()
        yield client


@pytest.fixture
def mock_requests_session():
    """Mock requests session for testing."""
    with patch("shared_ollama_client.requests.Session") as mock_session:
        session = Mock()
        mock_session.return_value = session
        yield session


@pytest.fixture
def sample_models_response():
    """Sample models API response."""
    return {
        "models": [
            {
                "name": "qwen2.5vl:7b",
                "model": "qwen2.5vl:7b",
                "modified_at": "2025-11-03T17:24:58.744838946Z",
                "size": 5969245856,
                "digest": "5ced39dfa4bac325dc183dd1e4febaa1c46b3ea28bce48896c8e69c1e79611cc",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen25vl",
                    "families": ["qwen25vl"],
                    "parameter_size": "8.3B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {
                "name": "qwen2.5:14b",
                "model": "qwen2.5:14b",
                "modified_at": "2025-11-03T14:30:29.181812332Z",
                "size": 8988124069,
                "digest": "7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "14.8B",
                    "quantization_level": "Q4_K_M",
                },
            },
        ]
    }


@pytest.fixture
def sample_generate_response():
    """Sample generate API response."""
    return {
        "model": "qwen2.5vl:7b",
        "created_at": "2025-11-05T11:00:00.000Z",
        "response": "Hello! How can I help you today?",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 500000000,
        "load_duration": 2000000000,
        "prompt_eval_count": 5,
        "prompt_eval_duration": 100000000,
        "eval_count": 10,
        "eval_duration": 400000000,
    }


@pytest.fixture
def sample_chat_response():
    """Sample chat API response."""
    return {
        "model": "qwen2.5vl:7b",
        "created_at": "2025-11-05T11:00:00.000Z",
        "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
        "done": True,
    }
