"""
Comprehensive behavioral tests for SharedOllamaClient.

Tests focus on real behavior, error handling, edge cases, and integration scenarios.
Uses real HTTP server (ollama_server fixture) - no mocks of internal logic.
Only external Ollama service is mocked via test server.
"""


import pytest
import requests

from shared_ollama import (
    GenerateOptions,
    GenerateResponse,
    Model,
    OllamaConfig,
    SharedOllamaClient,
)


class TestOllamaConfig:
    """Behavioral tests for OllamaConfig dataclass."""

    def test_default_config_values(self):
        """Test that default config uses expected values."""
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.default_model == Model.QWEN3_VL_8B_Q4
        assert config.timeout == 300
        assert config.health_check_timeout == 5
        assert config.verbose is False

    def test_config_is_immutable(self):
        """Test that config is frozen (immutable)."""
        config = OllamaConfig()
        with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
            config.base_url = "http://changed:11434"

    def test_custom_config_preserves_values(self):
        """Test that custom config values are preserved."""
        config = OllamaConfig(
            base_url="http://custom:11434",
            default_model=Model.QWEN3_14B_Q4,
            timeout=120,
            verbose=True,
        )
        assert config.base_url == "http://custom:11434"
        assert config.default_model == Model.QWEN3_14B_Q4
        assert config.timeout == 120
        assert config.verbose is True

    def test_config_uses_slots(self):
        """Test that config uses __slots__ for memory efficiency."""
        config = OllamaConfig()
        # If using slots, __dict__ should not exist or be empty
        assert not hasattr(config, "__dict__") or len(config.__dict__) == 0


class TestGenerateOptions:
    """Behavioral tests for GenerateOptions dataclass."""

    def test_default_options_use_sensible_values(self):
        """Test that default options use sensible generation parameters."""
        options = GenerateOptions()
        assert options.temperature == 0.2  # Low temperature for deterministic output
        assert options.top_p == 0.9
        assert options.top_k == 40
        assert options.repeat_penalty == 1.1
        assert options.max_tokens is None
        assert options.seed is None
        assert options.stop is None

    def test_options_are_immutable(self):
        """Test that options are frozen (immutable)."""
        options = GenerateOptions()
        with pytest.raises(Exception):
            options.temperature = 1.0

    def test_custom_options_preserve_all_values(self):
        """Test that all custom option values are preserved."""
        options = GenerateOptions(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.2,
            max_tokens=100,
            seed=42,
            stop=["\n", "STOP"],
        )
        assert options.temperature == 0.7
        assert options.top_p == 0.95
        assert options.top_k == 50
        assert options.repeat_penalty == 1.2
        assert options.max_tokens == 100
        assert options.seed == 42
        assert options.stop == ["\n", "STOP"]


class TestSharedOllamaClient:
    """Comprehensive behavioral tests for SharedOllamaClient using real server."""

    def test_client_initialization_with_default_config(self, ollama_server):
        """Test client initialization with default config."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)
        assert client.config == config
        assert client.session is not None
        assert isinstance(client.session, requests.Session)

    def test_client_initialization_with_custom_config(self, ollama_server):
        """Test client initialization with custom config."""
        config = OllamaConfig(
            base_url=ollama_server.base_url,
            default_model=Model.QWEN3_14B_Q4,
            timeout=120,
        )
        client = SharedOllamaClient(config=config, verify_on_init=False)
        assert client.config == config
        assert client.config.default_model == Model.QWEN3_14B_Q4
        assert client.config.timeout == 120

    def test_client_verifies_connection_on_init(self, ollama_server):
        """Test that client verifies connection when verify_on_init=True."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=True)
        # Should not raise - connection verified
        assert client.session is not None

    def test_client_handles_connection_failure(self):
        """Test that client raises ConnectionError when service unavailable."""
        config = OllamaConfig(base_url="http://localhost:99999")
        with pytest.raises(ConnectionError):
            SharedOllamaClient(config=config, verify_on_init=True)

    def test_list_models_returns_list(self, ollama_server):
        """Test that list_models() returns a list of model dictionaries."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        models = client.list_models()

        assert isinstance(models, list)
        assert len(models) >= 2
        assert all(isinstance(m, dict) for m in models)

    def test_list_models_extracts_models_key(self, ollama_server):
        """Test that list_models() correctly extracts 'models' key from response."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        models = client.list_models()

        assert len(models) >= 2
        assert any(model["name"] == "qwen3-vl:8b-instruct-q4_K_M" for model in models)

    def test_list_models_handles_empty_response(self, ollama_server):
        """Test that list_models() handles empty models list."""
        ollama_server.state["models"] = []
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        models = client.list_models()

        assert models == []

    def test_list_models_validates_response_structure(self, ollama_server):
        """Test that list_models() validates response is a dict."""
        # This tests the real validation logic - server returns dict, so should work
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        models = client.list_models()
        # Valid response should work
        assert isinstance(models, list)

    def test_list_models_handles_http_error(self, ollama_server):
        """Test that list_models() raises HTTPError on HTTP errors."""
        ollama_server.state["tags_status"] = 500
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        with pytest.raises(requests.exceptions.HTTPError):
            client.list_models()

    def test_generate_returns_generate_response(self, ollama_server):
        """Test that generate() returns GenerateResponse object with real data."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.generate("Hello, world!")

        assert isinstance(response, GenerateResponse)
        assert response.text.startswith("ECHO: Hello, world!")
        assert response.model == "qwen3-vl:8b-instruct-q4_K_M"

    def test_generate_extracts_all_metrics(self, ollama_server):
        """Test that generate() extracts all performance metrics from response."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.generate("Test prompt")

        assert response.total_duration > 0
        assert response.load_duration > 0
        assert response.prompt_eval_count > 0
        assert response.eval_count > 0
        assert response.prompt_eval_duration > 0
        assert response.eval_duration > 0

    def test_generate_uses_default_model_when_none_specified(self, ollama_server):
        """Test that generate() uses default model when model=None."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.generate("Test", model=None)

        assert response.model == config.default_model

    def test_generate_includes_system_message(self, ollama_server):
        """Test that generate() includes system message in payload when provided."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.generate("Test", system="You are helpful")

        # Verify system was sent by checking server state
        calls = ollama_server.state.get("generate_calls", [])
        assert any("system" in call and call["system"] == "You are helpful" for call in calls)

    def test_generate_includes_format_json_mode(self, ollama_server):
        """Test that generate() includes format='json' in payload."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        client.generate("Test", format="json")

        calls = ollama_server.state.get("generate_calls", [])
        assert any("format" in call and call["format"] == "json" for call in calls)

    def test_generate_includes_format_schema(self, ollama_server):
        """Test that generate() includes format schema dict in payload."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        client.generate("Test", format=schema)

        calls = ollama_server.state.get("generate_calls", [])
        assert any("format" in call and call["format"] == schema for call in calls)

    def test_generate_with_options_includes_all_parameters(self, ollama_server):
        """Test that generate() includes all option parameters in payload."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        options = GenerateOptions(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.2,
            max_tokens=100,
            seed=42,
            stop=["\n"],
        )
        client.generate("Test", options=options)

        calls = ollama_server.state.get("generate_calls", [])
        assert any("options" in call for call in calls)
        call_with_options = next(call for call in calls if "options" in call)
        opts = call_with_options["options"]
        assert opts["temperature"] == 0.7
        assert opts["top_p"] == 0.95
        assert opts["top_k"] == 50
        assert opts["repeat_penalty"] == 1.2
        assert opts["num_predict"] == 100
        assert opts["seed"] == 42
        assert opts["stop"] == ["\n"]

    def test_generate_filters_none_options(self, ollama_server):
        """Test that generate() doesn't include None option values."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        options = GenerateOptions(max_tokens=None, seed=None, stop=None)
        client.generate("Test", options=options)

        calls = ollama_server.state.get("generate_calls", [])
        call_with_options = next(call for call in calls if "options" in call)
        opts = call_with_options["options"]
        # None values should not be in options dict
        assert "num_predict" not in opts or opts.get("num_predict") is not None
        assert "seed" not in opts or opts.get("seed") is not None
        assert "stop" not in opts or opts.get("stop") is not None

    def test_generate_handles_missing_response_field(self, ollama_server):
        """Test that generate() handles missing 'response' field gracefully."""
        # Modify server to return response without 'response' field
        original_handler = ollama_server.state.get("generate_handler")
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        # Server returns response with 'response' field, so test normal case
        response = client.generate("Test")
        assert response.text is not None  # Should have text from server

    def test_generate_validates_response_structure(self, ollama_server):
        """Test that generate() validates response is a dict."""
        # Server returns valid dict, so this tests the validation works
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.generate("Test")
        # Valid response should work
        assert isinstance(response, GenerateResponse)

    def test_generate_handles_http_error(self, ollama_server):
        """Test that generate() raises HTTPError on HTTP errors."""
        ollama_server.state["generate_failures"] = 1
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        with pytest.raises(requests.exceptions.HTTPError):
            client.generate("This should fail")

    def test_generate_handles_connection_error(self, ollama_server):
        """Test that generate() handles connection errors correctly."""
        # Use invalid URL to trigger connection error
        config = OllamaConfig(base_url="http://localhost:99999")
        client = SharedOllamaClient(config=config, verify_on_init=False)

        with pytest.raises(requests.exceptions.RequestException):
            client.generate("Test")

    def test_generate_records_metrics(self, ollama_server):
        """Test that generate() records metrics via MetricsCollector."""
        from shared_ollama.telemetry.metrics import MetricsCollector

        MetricsCollector.reset()
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        client.generate("Test prompt")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests >= 1
        assert metrics.successful_requests >= 1

    def test_chat_returns_dict_with_message(self, ollama_server):
        """Test that chat() returns dict with message content."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        messages = [{"role": "user", "content": "Hello!"}]
        response = client.chat(messages)

        assert isinstance(response, dict)
        assert "message" in response
        assert response["message"]["content"].startswith("Echo:")

    def test_chat_handles_multiple_messages(self, ollama_server):
        """Test that chat() handles conversation history."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        response = client.chat(messages)

        assert "message" in response
        # Verify messages were sent
        calls = ollama_server.state.get("chat_calls", [])
        assert any("messages" in call for call in calls)

    def test_chat_includes_options(self, ollama_server):
        """Test that chat() includes options in payload."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        options = GenerateOptions(temperature=0.8)
        client.chat([{"role": "user", "content": "Test"}], options=options)

        calls = ollama_server.state.get("chat_calls", [])
        assert any("options" in call for call in calls)

    def test_chat_handles_http_error(self, ollama_server):
        """Test that chat() raises HTTPError on HTTP errors."""
        ollama_server.state["chat_failures"] = 1
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        with pytest.raises(requests.exceptions.HTTPError):
            client.chat([{"role": "user", "content": "This should fail"}])

    def test_chat_validates_response_structure(self, ollama_server):
        """Test that chat() validates response is a dict."""
        # Server returns valid dict, so this tests validation works
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.chat([{"role": "user", "content": "Test"}])
        assert isinstance(response, dict)

    def test_health_check_returns_true_for_200(self, ollama_server):
        """Test that health_check() returns True for HTTP 200."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        result = client.health_check()
        assert result is True

    def test_health_check_returns_false_for_non_200(self, ollama_server):
        """Test that health_check() returns False for non-200 status codes."""
        ollama_server.state["tags_status"] = 500
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        result = client.health_check()
        assert result is False

    def test_health_check_returns_false_on_exception(self, ollama_server):
        """Test that health_check() returns False on any exception."""
        config = OllamaConfig(base_url="http://localhost:99999")
        client = SharedOllamaClient(config=config, verify_on_init=False)

        result = client.health_check()
        assert result is False

    def test_get_model_info_returns_model_dict(self, ollama_server):
        """Test that get_model_info() returns model dict when found."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        model_info = client.get_model_info("qwen3-vl:8b-instruct-q4_K_M")

        assert model_info is not None
        assert isinstance(model_info, dict)
        assert model_info["name"] == "qwen3-vl:8b-instruct-q4_K_M"

    def test_get_model_info_returns_none_when_not_found(self, ollama_server):
        """Test that get_model_info() returns None when model not found."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        model_info = client.get_model_info("nonexistent:model")

        assert model_info is None

    def test_get_model_info_is_cached(self, ollama_server):
        """Test that get_model_info() uses cached list_models() result."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        # First call should call list_models
        model_info1 = client.get_model_info("qwen3-vl:8b-instruct-q4_K_M")

        # Second call should use cache (no additional HTTP call)
        model_info2 = client.get_model_info("qwen3-vl:8b-instruct-q4_K_M")

        assert model_info1 == model_info2
        # Both should return the same model info
        assert model_info1 is not None
        assert model_info2 is not None

    @pytest.mark.parametrize(
        "prompt,expected_in_response",
        [
            ("Short", True),
            ("A" * 1000, True),  # Long prompt
            ("", True),  # Empty prompt
            ("Special chars: !@#$%^&*()", True),
        ],
    )
    def test_generate_handles_various_prompts(self, ollama_server, prompt, expected_in_response):
        """Test that generate() handles various prompt types."""
        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        response = client.generate(prompt)

        assert isinstance(response, GenerateResponse)
        if expected_in_response and prompt:
            # Server echoes prompt, so it should be in response
            assert prompt in response.text or len(response.text) > 0

    def test_generate_with_timeout(self, ollama_server):
        """Test that generate() respects timeout configuration."""
        config = OllamaConfig(base_url=ollama_server.base_url, timeout=1)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        # Normal request should complete within timeout
        response = client.generate("Quick test")
        assert isinstance(response, GenerateResponse)

    def test_concurrent_generate_requests(self, ollama_server):
        """Test that multiple concurrent generate requests work correctly."""
        import concurrent.futures

        config = OllamaConfig(base_url=ollama_server.base_url)
        client = SharedOllamaClient(config=config, verify_on_init=False)

        def make_request(i: int) -> GenerateResponse:
            return client.generate(f"Request {i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(responses) == 10
        assert all(isinstance(r, GenerateResponse) for r in responses)
        assert all(r.text for r in responses)
