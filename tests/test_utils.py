"""
Unit tests for utility functions.
"""

import os
from unittest.mock import Mock, patch

import pytest
import requests

from shared_ollama_client import SharedOllamaClient
from utils import (
    check_service_health,
    ensure_service_running,
    get_client_path,
    get_ollama_base_url,
    get_project_root,
    import_client,
)


class TestGetOllamaBaseUrl:
    """Tests for get_ollama_base_url()."""

    def test_default_url(self):
        """Test default URL when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://localhost:11434"

    def test_explicit_base_url(self):
        """Test OLLAMA_BASE_URL environment variable."""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434"}):
            url = get_ollama_base_url()
            assert url == "http://custom:11434"

    def test_host_port_separate(self):
        """Test OLLAMA_HOST and OLLAMA_PORT environment variables."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "custom", "OLLAMA_PORT": "8080"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://custom:8080"

    def test_host_with_port(self):
        """Test OLLAMA_HOST with embedded port."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "custom:8080"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://custom:8080"


class TestCheckServiceHealth:
    """Tests for check_service_health()."""

    @patch("utils.requests.get")
    def test_healthy_service(self, mock_get):
        """Test healthy service check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        is_healthy, error = check_service_health()

        assert is_healthy is True
        assert error is None

    @patch("utils.requests.get")
    def test_unhealthy_service(self, mock_get):
        """Test unhealthy service check."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None and "status code 500" in error

    @patch("utils.requests.get")
    def test_connection_error(self, mock_get):
        """Test connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None and "Cannot connect" in error

    @patch("utils.requests.get")
    def test_timeout_error(self, mock_get):
        """Test timeout error."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None and "timed out" in error


class TestEnsureServiceRunning:
    """Tests for ensure_service_running()."""

    @patch("utils.check_service_health")
    def test_service_running(self, mock_check):
        """Test when service is running."""
        mock_check.return_value = (True, None)

        result = ensure_service_running(raise_on_fail=False)

        assert result is True

    @patch("utils.check_service_health")
    def test_service_not_running_no_raise(self, mock_check):
        """Test when service is not running without raising."""
        mock_check.return_value = (False, "Service unavailable")

        result = ensure_service_running(raise_on_fail=False)

        assert result is False

    @patch("utils.check_service_health")
    def test_service_not_running_with_raise(self, mock_check):
        """Test when service is not running with raising."""
        mock_check.return_value = (False, "Service unavailable")

        with pytest.raises(ConnectionError):
            ensure_service_running(raise_on_fail=True)


class TestProjectRoot:
    """Tests for get_project_root()."""

    def test_get_project_root(self):
        """Test getting project root."""
        root = get_project_root()
        assert root is not None
        assert root.exists()
        assert (root / "shared_ollama_client.py").exists()


class TestGetClientPath:
    """Tests for get_client_path()."""

    def test_get_client_path(self):
        """Test getting client path."""
        path = get_client_path()
        assert path is not None
        assert path.exists()
        assert path.name == "shared_ollama_client.py"


class TestImportClient:
    """Tests for import_client()."""

    def test_import_client(self):
        """Test importing client dynamically."""
        client_class = import_client()
        assert client_class is not None
        assert client_class == SharedOllamaClient
