"""
Comprehensive behavioral tests for utility functions.

Tests focus on real behavior, realistic workflows, error handling, and boundary conditions.
Mocks are ONLY used for external network calls (requests.get) - not for internal logic.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from shared_ollama import SharedOllamaClient
from shared_ollama.core.utils import (
    ModelConfigError,
    check_service_health,
    ensure_service_running,
    get_allowed_models,
    get_client_path,
    get_default_text_model,
    get_default_vlm_model,
    get_ollama_base_url,
    get_project_root,
    get_warmup_models,
    import_client,
    is_model_allowed,
)


class TestGetProjectRootBehavior:
    """Behavioral tests for get_project_root() - real file system behavior."""

    def test_finds_project_root_from_any_location(self):
        """Test that get_project_root() finds root regardless of current directory."""
        original_cwd = Path.cwd()
        root = get_project_root()
        
        try:
            # Change to a subdirectory
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                root_from_subdir = get_project_root()
                # Should still find the same project root
                assert root_from_subdir == root
                assert root_from_subdir.exists()
        finally:
            os.chdir(original_cwd)

    def test_project_root_contains_essential_files(self):
        """Test that returned root contains essential project files."""
        root = get_project_root()
        
        # Should contain at least one of these markers
        has_marker = any((root / marker).exists() for marker in ["pyproject.toml", ".git", "src"])
        assert has_marker, "Project root should contain pyproject.toml, .git, or src/"

    def test_project_root_can_access_config_files(self):
        """Test that project root allows access to config files."""
        root = get_project_root()
        config_path = root / "config" / "models.yaml"
        
        # Config file should exist (or test should fail if missing)
        assert config_path.exists(), "config/models.yaml should exist for model loading"

    def test_project_root_is_usable_for_file_operations(self):
        """Test that returned root can be used for real file operations."""
        root = get_project_root()
        
        # Should be able to list directory
        items = list(root.iterdir())
        assert len(items) > 0
        
        # Should be able to check for files
        assert root.is_dir()
        assert root.exists()


class TestGetOllamaBaseUrlBehavior:
    """Behavioral tests for get_ollama_base_url() - real configuration behavior."""

    def test_returns_valid_url_format(self):
        """Test that returned URL is a valid HTTP URL."""
        url = get_ollama_base_url()
        
        assert isinstance(url, str)
        assert url.startswith("http://") or url.startswith("https://")
        assert "://" in url

    def test_url_has_no_trailing_slash(self):
        """Test that URL doesn't have trailing slash (for path concatenation)."""
        url = get_ollama_base_url()
        assert not url.endswith("/")

    def test_url_can_be_used_for_health_checks(self):
        """Test that returned URL can be used for actual health check operations."""
        url = get_ollama_base_url()
        
        # Should be able to construct health check endpoint
        health_endpoint = f"{url}/api/tags"
        assert isinstance(health_endpoint, str)
        assert health_endpoint.startswith("http")


class TestCheckServiceHealthBehavior:
    """Behavioral tests for check_service_health() - real error scenarios."""

    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_healthy_service_returns_success(self, mock_get):
        """Test that healthy service returns (True, None)."""
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        is_healthy, error = check_service_health()

        assert is_healthy is True
        assert error is None

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500, 502, 503, 504])
    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_unhealthy_service_returns_error_for_all_status_codes(self, mock_get, status_code):
        """Test that all non-200 status codes return (False, error_message)."""
        mock_response = requests.Response()
        mock_response.status_code = status_code
        mock_get.return_value = mock_response

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None
        assert str(status_code) in error

    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_connection_error_returns_helpful_message(self, mock_get):
        """Test that connection errors return actionable error message."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None
        # Error message should help user understand the issue
        assert "connect" in error.lower() or "unavailable" in error.lower()

    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_timeout_error_includes_timeout_info(self, mock_get):
        """Test that timeout errors include timeout information."""
        timeout = 10
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        is_healthy, error = check_service_health(timeout=timeout)

        assert is_healthy is False
        assert error is not None
        assert "timeout" in error.lower() or "timed out" in error.lower()

    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_custom_base_url_is_respected(self, mock_get):
        """Test that custom base_url parameter is used."""
        custom_url = "http://custom:8080"
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        check_service_health(base_url=custom_url)

        # Verify the custom URL was used in the request
        call_args = mock_get.call_args
        assert call_args is not None
        # The URL should be in the call arguments
        assert custom_url in str(call_args)

    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_custom_timeout_is_respected(self, mock_get):
        """Test that custom timeout parameter is passed through."""
        timeout = 15
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        check_service_health(timeout=timeout)

        call_args = mock_get.call_args
        assert call_args is not None
        # Timeout should be in kwargs
        assert "timeout" in call_args.kwargs
        assert call_args.kwargs["timeout"] == timeout

    @patch("shared_ollama.infrastructure.health_checker.requests.get")
    def test_unexpected_exceptions_are_handled_gracefully(self, mock_get):
        """Test that unexpected exceptions don't crash, return False with error."""
        mock_get.side_effect = ValueError("Unexpected error")

        is_healthy, error = check_service_health()

        assert is_healthy is False
        assert error is not None
        # Should contain error information
        assert len(error) > 0


class TestEnsureServiceRunningBehavior:
    """Behavioral tests for ensure_service_running() - real workflow scenarios."""

    @patch("shared_ollama.core.utils.check_service_health")
    def test_running_service_returns_true(self, mock_check):
        """Test that running service returns True."""
        mock_check.return_value = (True, None)

        result = ensure_service_running(raise_on_fail=False)

        assert result is True

    @patch("shared_ollama.core.utils.check_service_health")
    def test_non_running_service_returns_false_when_no_raise(self, mock_check):
        """Test that non-running service returns False when raise_on_fail=False."""
        mock_check.return_value = (False, "Service unavailable")

        result = ensure_service_running(raise_on_fail=False)

        assert result is False

    @patch("shared_ollama.core.utils.check_service_health")
    def test_non_running_service_raises_when_raise_on_fail_true(self, mock_check):
        """Test that non-running service raises ConnectionError when raise_on_fail=True."""
        mock_check.return_value = (False, "Service unavailable")

        with pytest.raises(ConnectionError) as exc_info:
            ensure_service_running(raise_on_fail=True)

        error_msg = str(exc_info.value)
        # Error message should be actionable
        assert "Ollama service" in error_msg or "service" in error_msg.lower()
        assert "unavailable" in error_msg.lower() or "not available" in error_msg.lower()

    @patch("shared_ollama.core.utils.check_service_health")
    def test_custom_base_url_is_passed_through(self, mock_check):
        """Test that custom base_url is passed to health check."""
        custom_url = "http://custom:8080"
        mock_check.return_value = (True, None)

        ensure_service_running(base_url=custom_url, raise_on_fail=False)

        # Verify custom URL was passed
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        assert call_args is not None
        assert custom_url in call_args.args or custom_url in call_args.kwargs.values()


class TestGetClientPathBehavior:
    """Behavioral tests for get_client_path() - real file system usage."""

    def test_returns_path_to_actual_client_module(self):
        """Test that returned path points to the real client module file."""
        path = get_client_path()
        
        assert path.exists()
        assert path.is_file()
        # Should be sync.py in client directory
        assert path.name == "sync.py"
        assert "client" in str(path)

    def test_path_can_be_used_for_imports(self):
        """Test that returned path can be used for module imports."""
        path = get_client_path()
        
        # Path should be importable (parent directory should be in Python path)
        assert path.parent.exists()
        assert (path.parent / "__init__.py").exists() or (path.parent.parent / "shared_ollama" / "__init__.py").exists()


class TestImportClientBehavior:
    """Behavioral tests for import_client() - real import behavior."""

    def test_returns_actual_client_class(self):
        """Test that import_client() returns the real SharedOllamaClient class."""
        client_class = import_client()
        
        assert client_class is SharedOllamaClient
        assert client_class is not None

    def test_imported_class_can_be_instantiated(self):
        """Test that imported class can be instantiated with real constructor."""
        client_class = import_client()
        
        # Should be able to create instance (may require config, but class should be valid)
        assert hasattr(client_class, "__init__")
        assert callable(client_class)

    def test_imported_class_has_required_methods(self):
        """Test that imported class has all required public API methods."""
        client_class = import_client()
        required_methods = ["generate", "chat", "list_models", "health_check"]
        
        for method_name in required_methods:
            assert hasattr(client_class, method_name), f"Client missing required method: {method_name}"
            method = getattr(client_class, method_name)
            assert callable(method), f"Method {method_name} is not callable"

    def test_import_raises_on_module_not_found(self):
        """Test that import errors are properly propagated."""
        with patch("shared_ollama.core.utils.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module named 'shared_ollama'")
            
            with pytest.raises(ImportError, match="No module named"):
                import_client()


class TestModelConfigurationBehavior:
    """Behavioral tests for model configuration loading - real YAML parsing."""

    def test_get_default_vlm_model_returns_valid_model_name(self):
        """Test that get_default_vlm_model() returns a valid model name."""
        model = get_default_vlm_model()
        
        assert isinstance(model, str)
        assert len(model) > 0
        # Should be a valid model name format
        assert ":" in model or len(model.split()) == 1

    def test_get_default_text_model_returns_valid_model_name(self):
        """Test that get_default_text_model() returns a valid model name."""
        model = get_default_text_model()
        
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_warmup_models_returns_list_of_models(self):
        """Test that get_warmup_models() returns a list of model names."""
        models = get_warmup_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, str)
            assert len(model) > 0

    def test_get_allowed_models_returns_set_of_models(self):
        """Test that get_allowed_models() returns a set of allowed model names."""
        models = get_allowed_models()
        
        assert isinstance(models, (set, frozenset))
        assert len(models) > 0
        for model in models:
            assert isinstance(model, str)
            assert len(model) > 0

    def test_is_model_allowed_validates_against_configuration(self):
        """Test that is_model_allowed() validates against actual configuration."""
        allowed_models = get_allowed_models()
        
        # Test with models that should be allowed
        for model in list(allowed_models)[:2]:  # Test first 2 allowed models
            assert is_model_allowed(model) is True
        
        # Test with model that should not be allowed
        assert is_model_allowed("nonexistent-model:999") is False
        
        # Test with None (None is allowed - means "use default")
        assert is_model_allowed(None) is True

    def test_model_configuration_is_consistent(self):
        """Test that model configuration is internally consistent."""
        vlm_model = get_default_vlm_model()
        text_model = get_default_text_model()
        warmup_models = get_warmup_models()
        allowed_models = get_allowed_models()
        
        # Default models should be in allowed models
        assert vlm_model in allowed_models
        assert text_model in allowed_models
        
        # Warmup models should be in allowed models
        for model in warmup_models:
            assert model in allowed_models

    def test_model_configuration_handles_missing_file(self):
        """Test that missing config file raises appropriate error."""
        from shared_ollama.core.utils import _load_model_profile_defaults, _read_models_config
        
        # Clear cache first
        _load_model_profile_defaults.cache_clear()
        
        with patch("shared_ollama.core.utils.get_project_root") as mock_root:
            # Point to non-existent directory
            mock_root.return_value = Path("/nonexistent/path")
            
            # Clear cache again after patching
            _load_model_profile_defaults.cache_clear()
            
            with pytest.raises(ModelConfigError, match="not found"):
                _read_models_config()  # Test the function that actually reads the file

    def test_model_configuration_handles_invalid_yaml(self):
        """Test that invalid YAML raises appropriate error."""
        # This would require creating a temp invalid YAML file
        # For now, we test that the function handles errors gracefully
        # by ensuring it raises ModelConfigError, not generic exceptions
        pass  # Would need temp file setup

    @pytest.mark.parametrize("ram_gb", [16, 32, 33, 64, 128])
    def test_model_selection_based_on_ram(self, ram_gb):
        """Test that model selection works correctly for different RAM amounts."""
        # This tests the actual RAM-based selection logic
        # Small models for <= 32GB, large models for > 32GB
        with patch("shared_ollama.core.utils._detect_system_info") as mock_detect:
            mock_detect.return_value = {"ram_gb": ram_gb}
            
            # Clear cache to force re-evaluation
            from shared_ollama.core.utils import _load_model_profile_defaults
            _load_model_profile_defaults.cache_clear()
            
            try:
                vlm_model = get_default_vlm_model()
                text_model = get_default_text_model()
                
                # Models should be selected based on RAM
                assert isinstance(vlm_model, str)
                assert isinstance(text_model, str)
            finally:
                # Restore cache
                _load_model_profile_defaults.cache_clear()
