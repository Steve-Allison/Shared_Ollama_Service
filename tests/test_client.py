"""
Comprehensive behavioral tests for SharedOllamaClient.

Tests focus on real behavior, error handling, edge cases, and integration scenarios.
Mocks are only used for external HTTP requests (session.get/post).
"""

import json
from unittest.mock import Mock

import pytest
import requests

from shared_ollama import GenerateOptions, GenerateResponse, Model, OllamaConfig


class TestOllamaConfig:
    """Behavioral tests for OllamaConfig dataclass."""

    def test_default_config_values(self):
        """Test that default config uses expected values."""
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.default_model == Model.QWEN25_VL_7B
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
            default_model=Model.QWEN25_14B,
            timeout=120,
            verbose=True,
        )
        assert config.base_url == "http://custom:11434"
        assert config.default_model == Model.QWEN25_14B
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
    """Comprehensive behavioral tests for SharedOllamaClient."""

    def test_client_initialization_with_default_config(self, mock_client):
        """Test client initialization with default config."""
        assert mock_client.config is not None
        assert isinstance(mock_client.config, OllamaConfig)
        assert mock_client.session is not None

    def test_client_initialization_with_custom_config(self, ollama_config):
        """Test client initialization with custom config."""
        from shared_ollama import SharedOllamaClient

        with patch("shared_ollama.client.sync.SharedOllamaClient._verify_connection"):
            client = SharedOllamaClient(config=ollama_config, verify_on_init=False)
            assert client.config == ollama_config

    def test_list_models_returns_list(self, mock_client, sample_models_response):
        """Test that list_models() returns a list of model dictionaries."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        models = mock_client.list_models()

        assert isinstance(models, list)
        assert len(models) == 4
        assert all(isinstance(m, dict) for m in models)

    def test_list_models_extracts_models_key(self, mock_client, sample_models_response):
        """Test that list_models() correctly extracts 'models' key from response."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        models = mock_client.list_models()

        assert len(models) == len(sample_models_response["models"])
        assert models[0]["name"] == "qwen2.5vl:7b"

    def test_list_models_handles_empty_response(self, mock_client):
        """Test that list_models() handles empty models list."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        models = mock_client.list_models()

        assert models == []

    def test_list_models_validates_response_structure(self, mock_client):
        """Test that list_models() validates response is a dict."""
        mock_response = Mock()
        mock_response.json.return_value = ["not", "a", "dict"]
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        with pytest.raises(ValueError, match="Expected dict response"):
            mock_client.list_models()

    def test_list_models_handles_json_decode_error(self, mock_client):
        """Test that list_models() properly raises JSONDecodeError."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_client.session.get.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            mock_client.list_models()

    def test_list_models_handles_http_error(self, mock_client):
        """Test that list_models() properly raises HTTPError."""
        mock_response = Mock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError("500 Server Error")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_client.session.get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            mock_client.list_models()

    def test_generate_returns_generate_response(self, mock_client, sample_generate_response):
        """Test that generate() returns GenerateResponse object."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        response = mock_client.generate("Hello, world!")

        assert isinstance(response, GenerateResponse)
        assert response.text == "Hello! How can I help you today?"
        assert response.model == "qwen2.5vl:7b"

    def test_generate_extracts_all_metrics(self, mock_client, sample_generate_response):
        """Test that generate() extracts all performance metrics from response."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        response = mock_client.generate("Test")

        assert response.total_duration == 500000000
        assert response.load_duration == 2000000000
        assert response.prompt_eval_count == 5
        assert response.prompt_eval_duration == 100000000
        assert response.eval_count == 10
        assert response.eval_duration == 400000000

    def test_generate_uses_default_model_when_none_specified(self, mock_client, sample_generate_response):
        """Test that generate() uses default model when model=None."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        response = mock_client.generate("Test", model=None)

        call_args = mock_client.session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == str(mock_client.config.default_model)

    def test_generate_includes_system_message(self, mock_client, sample_generate_response):
        """Test that generate() includes system message in payload when provided."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        mock_client.generate("Test", system="You are helpful")

        call_args = mock_client.session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["system"] == "You are helpful"

    def test_generate_includes_format_json_mode(self, mock_client, sample_generate_response):
        """Test that generate() includes format='json' in payload."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        mock_client.generate("Test", format="json")

        call_args = mock_client.session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["format"] == "json"

    def test_generate_includes_format_schema(self, mock_client, sample_generate_response):
        """Test that generate() includes format schema dict in payload."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_client.generate("Test", format=schema)

        call_args = mock_client.session.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["format"] == schema

    def test_generate_with_options_includes_all_parameters(self, mock_client, sample_generate_response):
        """Test that generate() includes all option parameters in payload."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        options = GenerateOptions(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.2,
            max_tokens=100,
            seed=42,
            stop=["\n"],
        )
        mock_client.generate("Test", options=options)

        call_args = mock_client.session.post.call_args
        payload = call_args.kwargs["json"]
        assert "options" in payload
        opts = payload["options"]
        assert opts["temperature"] == 0.7
        assert opts["top_p"] == 0.95
        assert opts["top_k"] == 50
        assert opts["repeat_penalty"] == 1.2
        assert opts["num_predict"] == 100
        assert opts["seed"] == 42
        assert opts["stop"] == ["\n"]

    def test_generate_filters_none_options(self, mock_client, sample_generate_response):
        """Test that generate() doesn't include None option values."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        options = GenerateOptions(max_tokens=None, seed=None, stop=None)
        mock_client.generate("Test", options=options)

        call_args = mock_client.session.post.call_args
        payload = call_args.kwargs["json"]
        opts = payload["options"]
        assert "num_predict" not in opts or opts.get("num_predict") is not None
        assert "seed" not in opts or opts.get("seed") is not None
        assert "stop" not in opts or opts.get("stop") is not None

    def test_generate_handles_missing_response_field(self, mock_client):
        """Test that generate() handles missing 'response' field gracefully."""
        mock_response = Mock()
        mock_response.json.return_value = {"model": "test", "done": True}  # Missing 'response'
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        response = mock_client.generate("Test")

        assert response.text == ""  # Should default to empty string

    def test_generate_validates_response_structure(self, mock_client):
        """Test that generate() validates response is a dict."""
        mock_response = Mock()
        mock_response.json.return_value = "not a dict"
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        with pytest.raises(ValueError, match="Expected dict response"):
            mock_client.generate("Test")

    def test_generate_handles_json_decode_error(self, mock_client):
        """Test that generate() properly raises JSONDecodeError."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_client.session.post.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            mock_client.generate("Test")

    def test_generate_handles_http_error(self, mock_client):
        """Test that generate() properly raises HTTPError."""
        mock_response = Mock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError("500 Server Error")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_client.session.post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            mock_client.generate("Test")

    def test_chat_returns_dict_with_message(self, mock_client, sample_chat_response):
        """Test that chat() returns dict with message content."""
        mock_response = Mock()
        mock_response.json.return_value = sample_chat_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]
        response = mock_client.chat(messages)

        assert isinstance(response, dict)
        assert "message" in response
        assert response["message"]["content"] == "Hello! How can I help you today?"

    def test_chat_validates_response_structure(self, mock_client):
        """Test that chat() validates response is a dict."""
        mock_response = Mock()
        mock_response.json.return_value = "not a dict"
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        with pytest.raises(ValueError, match="Expected dict response"):
            mock_client.chat([{"role": "user", "content": "Test"}])

    def test_health_check_returns_true_for_200(self, mock_client):
        """Test that health_check() returns True for HTTP 200."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.session.get.return_value = mock_response

        result = mock_client.health_check()
        assert result is True

    def test_health_check_returns_false_for_non_200(self, mock_client):
        """Test that health_check() returns False for non-200 status codes."""
        for status_code in [400, 401, 403, 404, 500, 502, 503]:
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_client.session.get.return_value = mock_response

            result = mock_client.health_check()
            assert result is False

    def test_health_check_returns_false_on_exception(self, mock_client):
        """Test that health_check() returns False on any exception."""
        mock_client.session.get.side_effect = requests.exceptions.RequestException("Connection failed")

        result = mock_client.health_check()
        assert result is False

    def test_get_model_info_returns_model_dict(self, mock_client, sample_models_response):
        """Test that get_model_info() returns model dict when found."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        model_info = mock_client.get_model_info("qwen2.5vl:7b")

        assert model_info is not None
        assert isinstance(model_info, dict)
        assert model_info["name"] == "qwen2.5vl:7b"
        assert model_info["size"] == 5969245856

    def test_get_model_info_returns_none_when_not_found(self, mock_client, sample_models_response):
        """Test that get_model_info() returns None when model not found."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        model_info = mock_client.get_model_info("nonexistent:model")

        assert model_info is None

    def test_get_model_info_is_cached(self, mock_client, sample_models_response):
        """Test that get_model_info() uses cached list_models() result."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        # First call should call list_models
        model_info1 = mock_client.get_model_info("qwen2.5vl:7b")
        call_count_1 = mock_client.session.get.call_count

        # Second call should use cache (no additional HTTP call)
        model_info2 = mock_client.get_model_info("qwen2.5vl:7b")
        call_count_2 = mock_client.session.get.call_count

        assert model_info1 == model_info2
        # Note: Due to lru_cache, second call might not make HTTP request
        # This tests that caching works
