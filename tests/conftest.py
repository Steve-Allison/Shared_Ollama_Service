"""
Pytest configuration and fixtures for Shared Ollama Service tests.
"""

import json
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared_ollama import OllamaConfig, SharedOllamaClient


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class OllamaRequestHandler(BaseHTTPRequestHandler):
    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = json.dumps(data).encode("utf-8")
        self.wfile.write(payload)

    def do_GET(self):
        state = self.server.server_state  # type: ignore[attr-defined]
        if self.path == "/api/tags":
            status = state.get("tags_status", 200)
            if status != 200:
                self._json_response({"error": "tags unavailable"}, status=status)
                return
            self._json_response({"models": state["models"]})
            return

        if self.path == "/api/version":
            self._json_response({"version": "0.0-test"})
            return

        self._json_response({"error": "not found"}, status=404)

    def do_POST(self):
        state = self.server.server_state  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}

        if self.path == "/api/generate":
            failures = state.get("generate_failures", 0)
            if failures > 0:
                state["generate_failures"] = failures - 1
                self._json_response({"error": "temporary failure"}, status=500)
                return

            prompt = payload.get("prompt", "")
            model = payload.get("model", "qwen3-vl:32b")
            state.setdefault("generate_calls", []).append(payload)
            response = {
                "model": model,
                "response": f"ECHO: {prompt}",
                "context": [1, 2, 3],
                "total_duration": state.get("total_duration_ns", 500_000_000),
                "load_duration": state.get("load_duration_ns", 200_000_000),
                "prompt_eval_count": len(prompt),
                "prompt_eval_duration": 100_000_000,
                "eval_count": max(len(prompt) // 2, 1),
                "eval_duration": 300_000_000,
            }
            self._json_response(response)
            return

        if self.path == "/api/chat":
            failures = state.get("chat_failures", 0)
            if failures > 0:
                state["chat_failures"] = failures - 1
                self._json_response({"error": "chat failure"}, status=500)
                return

            messages = payload.get("messages", [])
            last = messages[-1]["content"] if messages else ""
            model = payload.get("model", "qwen3-vl:32b")
            state.setdefault("chat_calls", []).append(payload)
            self._json_response(
                {
                    "model": model,
                    "message": {"role": "assistant", "content": f"Echo: {last}"},
                    "done": True,
                }
            )
            return

        if self.path == "/api/pull":
            failures = state.get("pull_failures", 0)
            if failures > 0:
                state["pull_failures"] = failures - 1
                self._json_response({"status": "error"}, status=500)
                return

            state.setdefault("pull_calls", []).append(payload)
            self._json_response({"status": "success"})
            return

        self._json_response({"error": "not found"}, status=404)

    def log_message(self, format, *args):
        # Suppress default HTTP server logging to keep test output clean.
        return


@pytest.fixture
def ollama_server():
    """Start a lightweight HTTP server that mimics essential Ollama endpoints."""
    state = {
        "models": [
            {"name": "qwen3-vl:32b"},
            {"name": "qwen3-vl:32b"},
        ],
        "generate_failures": 0,
        "chat_failures": 0,
        "pull_failures": 0,
        "tags_status": 200,
        "load_duration_ns": 200_000_000,
        "total_duration_ns": 500_000_000,
    }

    handler = OllamaRequestHandler
    server = ThreadedTCPServer(("127.0.0.1", 0), handler)
    server.server_state = state  # type: ignore[attr-defined]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        yield SimpleNamespace(base_url=base_url, state=state)
    finally:
        server.shutdown()
        server.server_close()


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
                "name": "qwen3-vl:32b",
                "model": "qwen3-vl:32b",
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
                "name": "qwen3-vl:32b",
                "model": "qwen3-vl:32b",
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
                "name": "qwen3:30b",
                "model": "qwen3:30b",
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
                "name": "granite4:small-h",
                "model": "granite4:small-h",
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
        "model": "qwen3-vl:32b",
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
        "model": "qwen3-vl:32b",
        "created_at": "2025-11-05T11:00:00.000Z",
        "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
        "done": True,
    }


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test to ensure isolation."""
    # Only reset if modules are already imported and have reset method
    if "shared_ollama.telemetry.metrics" in sys.modules:
        from shared_ollama.telemetry.metrics import MetricsCollector
        MetricsCollector.reset()

    if "shared_ollama.telemetry.performance" in sys.modules:
        from shared_ollama.telemetry.performance import PerformanceCollector
        PerformanceCollector.reset()

    yield

    # Cleanup after test - only if modules are imported
    if "shared_ollama.telemetry.metrics" in sys.modules:
        from shared_ollama.telemetry.metrics import MetricsCollector
        MetricsCollector.reset()

    if "shared_ollama.telemetry.performance" in sys.modules:
        from shared_ollama.telemetry.performance import PerformanceCollector
        PerformanceCollector.reset()
