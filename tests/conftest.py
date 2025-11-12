"""
Pytest configuration and fixtures for Shared Ollama Service tests.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared_ollama import OllamaConfig, SharedOllamaClient


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
    with patch("shared_ollama.client.sync.SharedOllamaClient._verify_connection"):
        client = SharedOllamaClient(config=ollama_config, verify_on_init=False)
        # Ensure session is a Mock for testing
        client.session = Mock()
        yield client


@pytest.fixture
def mock_requests_session():
    """Mock requests session for testing."""
    with patch("shared_ollama.client.sync.requests.Session") as mock_session:
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
                "name": "qwen2.5:7b",
                "model": "qwen2.5:7b",
                "modified_at": "2025-11-03T15:00:00.000000000Z",
                "size": 4730000000,
                "digest": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "7B",
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
            {
                "name": "granite4:tiny-h",
                "model": "granite4:tiny-h",
                "modified_at": "2025-11-06T12:00:00.000000000Z",
                "size": 4500000000,
                "digest": "8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "granite4",
                    "families": ["granite4"],
                    "parameter_size": "7B",
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
