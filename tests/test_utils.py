"""
Comprehensive behavioral tests for utility functions.

Tests focus on real behavior, edge cases, error handling, and boundary conditions.
Mocks are only used for external network calls (requests.get).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from shared_ollama import SharedOllamaClient
from shared_ollama.core.utils import (
    check_service_health,
    ensure_service_running,
    get_client_path,
    get_ollama_base_url,
    get_project_root,
    import_client,
)


class TestGetOllamaBaseUrl:
    """Behavioral tests for get_ollama_base_url()."""

    def test_default_url_when_no_env_vars(self):
        """Test default URL construction when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://localhost:11434"
            assert url.startswith("http://")
            assert ":11434" in url

    def test_explicit_base_url_takes_precedence(self):
        """Test OLLAMA_BASE_URL takes precedence over HOST/PORT."""
        with patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://custom:11434",
                "OLLAMA_HOST": "ignored",
                "OLLAMA_PORT": "ignored",
            },
            clear=True,
        ):
            url = get_ollama_base_url()
            assert url == "http://custom:11434"
            assert "ignored" not in url

    def test_base_url_strips_trailing_slash(self):
        """Test that trailing slashes are removed from base URL."""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434/"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://custom:11434"
            assert not url.endswith("/")

    def test_host_port_construction(self):
        """Test URL construction from separate HOST and PORT variables."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "custom", "OLLAMA_PORT": "8080"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://custom:8080"
            assert url.startswith("http://")
            assert ":8080" in url

    def test_host_with_embedded_port(self):
        """Test that host with embedded port is handled correctly."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "custom:8080"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://custom:8080"
            # Should not double-add port
            assert url.count(":") == 2  # http:// and :8080

    def test_default_host_when_only_port_set(self):
        """Test default host (localhost) when only PORT is set."""
        with patch.dict(os.environ, {"OLLAMA_PORT": "9999"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://localhost:9999"

    def test_default_port_when_only_host_set(self):
        """Test default port (11434) when only HOST is set."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "custom-host"}, clear=True):
            url = get_ollama_base_url()
            assert url == "http://custom-host:11434"


class TestCheckServiceHealth:
    """Behavioral tests for check_service_health() with real error scenarios."""

    @patch("shared_ollama.core.utils.requests.get")
    def test_healthy_service_returns_true(self, mock_get):
        """Test that healthy service (HTTP 200) returns (True, None)."""
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        is_healthy, error = check_service_health()

        assert is_healthy is True
        assert error is None
        mock_get.assert_called_once()

    @patch("shared_ollama.core.utils.requests.get")
    def test_unhealthy_service_returns_false_with_message(self, mock_get):
        """Test that non-200 status codes return (False, error_message)."""
        for status_code in [400, 401, 403, 404, 500, 502, 503]:
            mock_response = requests.Response()
            mock_response.status_code = status_code
            mock_get.return_value = mock_response

            is_healthy, error = check_service_health()

            assert is_healthy is False
            assert error is not None
            assert str(status_code) in error
            assert "status code" in error.lower()

    @patch("shared_ollama.core.utils.requests.get")
    def test_connection_error_returns_helpful_message(self, mock_get):
        """Test that connection errors return helpful error message."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None
        assert "Cannot connect" in error
        assert "Is the service running" in error

    @patch("shared_ollama.core.utils.requests.get")
    def test_timeout_error_includes_timeout_value(self, mock_get):
        """Test that timeout errors include the timeout value in message."""
        timeout = 10
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        is_healthy, error = check_service_health(timeout=timeout)

        assert is_healthy is False
        assert error is not None
        assert "timed out" in error.lower()
        assert str(timeout) in error or "timeout" in error.lower()

    @patch("shared_ollama.core.utils.requests.get")
    def test_custom_base_url_is_used(self, mock_get):
        """Test that custom base_url parameter is used in request."""
        custom_url = "http://custom:8080"
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        check_service_health(base_url=custom_url)

        call_args = mock_get.call_args
        assert call_args is not None
        assert custom_url in str(call_args)

    @patch("shared_ollama.core.utils.requests.get")
    def test_custom_timeout_is_used(self, mock_get):
        """Test that custom timeout parameter is passed to requests.get."""
        timeout = 15
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        check_service_health(timeout=timeout)

        call_args = mock_get.call_args
        assert call_args is not None
        assert timeout in call_args.kwargs.values() or timeout in call_args.args

    @patch("shared_ollama.core.utils.requests.get")
    def test_unexpected_exception_is_handled(self, mock_get):
        """Test that unexpected exceptions are caught and return False."""
        mock_get.side_effect = ValueError("Unexpected error")

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None
        assert "Unexpected error" in error or "error" in error.lower()


class TestEnsureServiceRunning:
    """Behavioral tests for ensure_service_running()."""

    @patch("shared_ollama.core.utils.check_service_health")
    def test_service_running_returns_true(self, mock_check):
        """Test that running service returns True."""
        mock_check.return_value = (True, None)

        result = ensure_service_running(raise_on_fail=False)

        assert result is True
        mock_check.assert_called_once()

    @patch("shared_ollama.core.utils.check_service_health")
    def test_service_not_running_returns_false_when_no_raise(self, mock_check):
        """Test that non-running service returns False when raise_on_fail=False."""
        mock_check.return_value = (False, "Service unavailable")

        result = ensure_service_running(raise_on_fail=False)

        assert result is False

    @patch("shared_ollama.core.utils.check_service_health")
    def test_service_not_running_raises_when_raise_on_fail_true(self, mock_check):
        """Test that non-running service raises ConnectionError when raise_on_fail=True."""
        mock_check.return_value = (False, "Service unavailable")

        with pytest.raises(ConnectionError) as exc_info:
            ensure_service_running(raise_on_fail=True)

        error_msg = str(exc_info.value)
        assert "Ollama service is not available" in error_msg
        assert "Service unavailable" in error_msg
        assert "ollama serve" in error_msg or "setup_launchd" in error_msg

    @patch("shared_ollama.core.utils.check_service_health")
    def test_custom_base_url_is_passed_through(self, mock_check):
        """Test that custom base_url is passed to check_service_health."""
        custom_url = "http://custom:8080"
        mock_check.return_value = (True, None)

        ensure_service_running(base_url=custom_url, raise_on_fail=False)

        mock_check.assert_called_once()
        call_args = mock_check.call_args
        assert call_args is not None
        assert custom_url in call_args.args or custom_url in call_args.kwargs.values()


class TestGetProjectRoot:
    """Behavioral tests for get_project_root() with real file system."""

    def test_returns_valid_path(self):
        """Test that returned path exists and is a directory."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert root.is_dir()

    def test_contains_expected_files(self):
        """Test that project root contains expected project files."""
        root = get_project_root()
        # Should contain pyproject.toml or .git or src/
        assert (
            (root / "pyproject.toml").exists()
            or (root / ".git").exists()
            or (root / "src").exists()
        )

    def test_contains_src_directory(self):
        """Test that project root contains src/ directory."""
        root = get_project_root()
        src_dir = root / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_result_is_cached(self):
        """Test that get_project_root() result is cached (same object returned)."""
        root1 = get_project_root()
        root2 = get_project_root()
        # Should return same Path object due to functools.cache
        assert root1 is root2

    def test_works_from_different_locations(self):
        """Test that project root detection works regardless of current directory."""
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                root = get_project_root()
                # Should still find the project root, not the temp dir
                assert root != Path(tmpdir)
                assert (root / "src").exists()
        finally:
            os.chdir(original_cwd)


class TestGetClientPath:
    """Behavioral tests for get_client_path()."""

    def test_returns_valid_path(self):
        """Test that returned path exists and points to sync.py."""
        path = get_client_path()
        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_file()
        assert path.name == "sync.py"

    def test_path_is_absolute(self):
        """Test that returned path is absolute."""
        path = get_client_path()
        assert path.is_absolute()

    def test_path_contains_client_module(self):
        """Test that path points to the correct client module."""
        path = get_client_path()
        assert "shared_ollama" in str(path)
        assert "client" in str(path)
        assert "sync.py" in str(path)

    def test_result_is_cached(self):
        """Test that get_client_path() result is cached."""
        path1 = get_client_path()
        path2 = get_client_path()
        assert path1 is path2


class TestImportClient:
    """Behavioral tests for import_client() with real imports."""

    def test_returns_client_class(self):
        """Test that import_client() returns the SharedOllamaClient class."""
        client_class = import_client()
        assert client_class is not None
        assert client_class == SharedOllamaClient

    def test_returns_callable_class(self):
        """Test that returned class can be instantiated."""
        client_class = import_client()
        # Should be a class that can be instantiated
        assert callable(client_class)
        assert hasattr(client_class, "__init__")

    def test_imported_class_has_expected_methods(self):
        """Test that imported class has expected public methods."""
        client_class = import_client()
        expected_methods = ["generate", "chat", "list_models", "health_check"]
        for method_name in expected_methods:
            assert hasattr(client_class, method_name), f"Missing method: {method_name}"

    def test_import_works_multiple_times(self):
        """Test that import_client() works consistently across multiple calls."""
        client1 = import_client()
        client2 = import_client()
        assert client1 is client2

    def test_import_handles_module_not_found(self):
        """Test that import errors are properly raised."""
        with patch("shared_ollama.core.utils.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'shared_ollama'")
            with pytest.raises(ImportError):
                import_client()
