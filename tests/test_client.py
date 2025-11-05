"""
Unit tests for SharedOllamaClient.
"""

from unittest.mock import Mock

from shared_ollama_client import (
    GenerateOptions,
    GenerateResponse,
    Model,
    OllamaConfig,
)


class TestOllamaConfig:
    """Tests for OllamaConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.default_model == Model.QWEN25_VL_7B
        assert config.timeout == 60
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration."""
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


class TestGenerateOptions:
    """Tests for GenerateOptions."""

    def test_default_options(self):
        """Test default generation options."""
        options = GenerateOptions()
        assert options.temperature == 0.2
        assert options.top_p == 0.9
        assert options.top_k == 40
        assert options.repeat_penalty == 1.1
        assert options.max_tokens is None
        assert options.seed is None
        assert options.stop is None

    def test_custom_options(self):
        """Test custom generation options."""
        options = GenerateOptions(
            temperature=0.7,
            top_p=0.95,
            max_tokens=100,
            seed=42,
            stop=["\n", "STOP"],
        )
        assert options.temperature == 0.7
        assert options.top_p == 0.95
        assert options.max_tokens == 100
        assert options.seed == 42
        assert options.stop == ["\n", "STOP"]


class TestSharedOllamaClient:
    """Tests for SharedOllamaClient."""

    def test_client_initialization(self, mock_client):
        """Test client initialization."""
        assert mock_client.config is not None
        assert mock_client.session is not None

    def test_list_models(self, mock_client, sample_models_response):
        """Test listing models."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        models = mock_client.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "qwen2.5vl:7b"
        assert models[1]["name"] == "qwen2.5:14b"
        mock_client.session.get.assert_called_once_with(
            f"{mock_client.config.base_url}/api/tags", timeout=mock_client.config.timeout
        )

    def test_generate(self, mock_client, sample_generate_response):
        """Test text generation."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        response = mock_client.generate("Hello, world!")

        assert isinstance(response, GenerateResponse)
        assert response.text == "Hello! How can I help you today?"
        assert response.model == "qwen2.5vl:7b"
        assert response.total_duration == 500000000
        mock_client.session.post.assert_called_once()

    def test_generate_with_options(self, mock_client, sample_generate_response):
        """Test text generation with custom options."""
        mock_response = Mock()
        mock_response.json.return_value = sample_generate_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        options = GenerateOptions(temperature=0.7, max_tokens=100)
        response = mock_client.generate("Hello!", options=options)

        assert isinstance(response, GenerateResponse)
        # Verify options were included in the request
        call_args = mock_client.session.post.call_args
        assert "options" in call_args.kwargs["json"]

    def test_chat(self, mock_client, sample_chat_response):
        """Test chat functionality."""
        mock_response = Mock()
        mock_response.json.return_value = sample_chat_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]
        response = mock_client.chat(messages)

        assert response["message"]["content"] == "Hello! How can I help you today?"
        mock_client.session.post.assert_called_once()

    def test_health_check_success(self, mock_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.session.get.return_value = mock_response

        assert mock_client.health_check() is True

    def test_health_check_failure(self, mock_client):
        """Test failed health check."""
        mock_client.session.get.side_effect = Exception("Connection failed")

        assert mock_client.health_check() is False

    def test_get_model_info(self, mock_client, sample_models_response):
        """Test getting model information."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        model_info = mock_client.get_model_info("qwen2.5vl:7b")

        assert model_info is not None
        assert model_info["name"] == "qwen2.5vl:7b"
        assert model_info["size"] == 5969245856

    def test_get_model_info_not_found(self, mock_client, sample_models_response):
        """Test getting model information for non-existent model."""
        mock_response = Mock()
        mock_response.json.return_value = sample_models_response
        mock_response.raise_for_status.return_value = None
        mock_client.session.get.return_value = mock_response

        model_info = mock_client.get_model_info("nonexistent:model")

        assert model_info is None
