"""Utility helpers for the Shared Ollama Service.

This module provides common utilities for path resolution, service health checks,
dynamic imports, and model configuration loading. All functions are designed to be
stateless and cacheable for performance.

Key behaviors:
    - Project root detection works for both editable installs and installed packages
    - Service health checks use timeouts and proper error handling
    - All path operations use pathlib for cross-platform compatibility
    - Model defaults loaded from config/model_profiles.yaml based on system hardware
"""

from __future__ import annotations

import functools
import importlib
import math
import os
import platform
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from shared_ollama.client.sync import SharedOllamaClient

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from shared_ollama.infrastructure.config import OllamaConfig
from shared_ollama.infrastructure.health_checker import check_ollama_health


@functools.cache
def get_project_root() -> Path:
    """Return the project root directory.

    Resolves the project root by walking up from the current module location
    until finding repository markers (pyproject.toml or .git). The result is
    cached for performance since the project root doesn't change at runtime.

    Returns:
        Path to project root directory.

    Raises:
        RuntimeError: If project root cannot be determined (should not occur
            in normal operation).
    """
    package_root = Path(__file__).resolve().parents[3]

    # Use match/case with guard for cleaner pattern matching (Python 3.13+)
    match (package_root / "pyproject.toml").exists():
        case True:
            return package_root
        case False:
            for parent in takewhile(
                lambda p: p != Path("/"),
                Path(__file__).resolve().parents,
            ):
                if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                    return parent
            return package_root


def get_ollama_base_url() -> str:
    """Get the Ollama base URL from environment variables.

    Uses pydantic-settings (via OllamaConfig) for consistent environment
    variable handling. Falls back to localhost:11434 if not set.

    Returns:
        Base URL string (e.g., "http://localhost:11434"). Always includes
        protocol and port, with trailing slashes removed.
    """
    # Use pydantic-settings for environment variable loading
    config = OllamaConfig()
    return config.url


def check_service_health(
    base_url: str | None = None,
    timeout: int = 5,
) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
    """Check if the Ollama service is healthy.

    Performs a lightweight health check by requesting the /api/tags endpoint.
    Returns a tuple indicating health status and optional error message.

    This function delegates to the infrastructure layer for HTTP operations,
    keeping the core module framework-agnostic.

    Args:
        base_url: Base URL for Ollama service. If None, uses get_ollama_base_url().
        timeout: Request timeout in seconds. Defaults to 5 seconds.

    Returns:
        Tuple of (is_healthy, error_message):
            - (True, None) if service is healthy
            - (False, str) if service is unhealthy, with error message

    Side effects:
        Makes an HTTP GET request to the Ollama service (via infrastructure layer).
    """
    if base_url is None:
        base_url = get_ollama_base_url()

    return check_ollama_health(base_url=base_url, timeout=timeout)


def ensure_service_running(
    base_url: str | None = None,
    raise_on_fail: bool = True,
) -> bool:
    """Ensure the Ollama service is running.

    Convenience wrapper around check_service_health() that can raise an exception
    if the service is not available. Useful for startup validation.

    Args:
        base_url: Base URL for Ollama service. If None, uses get_ollama_base_url().
        raise_on_fail: If True, raise ConnectionError when service is not available.
            If False, return False instead.

    Returns:
        True if service is running, False if raise_on_fail is False and service
        is not available.

    Raises:
        ConnectionError: If raise_on_fail is True and service is not available.
            Includes helpful error message with instructions.

    Side effects:
        Calls check_service_health(), which makes an HTTP request.
    """
    is_healthy, error = check_service_health(base_url)

    # Use match/case with guards for cleaner conditional logic (Python 3.13+)
    match (is_healthy, raise_on_fail):
        case (False, True):
            msg = (
                f"Ollama service is not available. {error}\n"
                "Start the service with: ./scripts/start.sh\n"
                "Or manually: ollama serve"
            )
            raise ConnectionError(msg)
        case (healthy, _):
            return healthy


@functools.cache
def get_client_path() -> Path:
    """Return the path to the synchronous client module.

    Resolves the absolute path to sync.py in the client package. The result
    is cached since the path doesn't change at runtime.

    Returns:
        Absolute Path to sync.py client module.
    """
    return (get_project_root() / "src" / "shared_ollama" / "client" / "sync.py").resolve()


def import_client() -> type[SharedOllamaClient]:
    """Dynamically import and return the SharedOllamaClient class.

    Uses importlib for runtime imports. Useful for lazy loading or when
    avoiding circular dependencies.

    Returns:
        The SharedOllamaClient class (not an instance).

    Raises:
        ImportError: If the module or class cannot be imported.
    """
    module = importlib.import_module("shared_ollama.client.sync")
    return module.SharedOllamaClient


@functools.cache
def _detect_system_info() -> tuple[str, int]:
    """Detect system architecture and RAM.

    Returns:
        Tuple of (arch, total_ram_gb).
    """
    arch = platform.machine()
    # Try to get total RAM
    total_ram_gb = 0
    try:
        if platform.system() == "Darwin":  # macOS
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                total_ram_gb = int(result.stdout.strip()) // (1024**3)
        elif platform.system() == "Linux":
            # Try /proc/meminfo
            try:
                with Path("/proc/meminfo").open() as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_gb = int(line.split()[1]) // (1024**2)
                            break
            except (OSError, ValueError):
                pass
    except Exception:
        pass

    return arch, total_ram_gb


@functools.cache
def _load_model_profile_defaults() -> dict[str, str | int | list[str]]:
    """Load model defaults from config/model_profiles.yaml based on system hardware.

    Selects the appropriate profile based on architecture and RAM, matching the
    logic in scripts/lib/model_config.sh. Falls back to safe defaults if profile
    loading fails.

    Returns:
        Dictionary with model defaults:
            - vlm_model: Default VLM model name
            - text_model: Default text model name
            - required_models: List of required model names
            - warmup_models: List of warmup model names
    """
    # First check environment variables (highest priority)
    env_vlm = os.getenv("OLLAMA_DEFAULT_VLM_MODEL")
    env_text = os.getenv("OLLAMA_DEFAULT_TEXT_MODEL")

    if env_vlm and env_text:
        return {
            "vlm_model": env_vlm,
            "text_model": env_text,
            "required_models": [env_vlm, env_text],
            "warmup_models": [env_vlm, env_text],
        }

    # Try to load from profile file
    defaults: dict[str, str | int | list[str]] = {}
    if yaml:
        try:
            project_root = get_project_root()
            profile_path = project_root / "config" / "model_profiles.yaml"
            if profile_path.exists():
                with profile_path.open(encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                profiles = data.get("profiles") or {}
                arch, ram = _detect_system_info()

                # Select matching profile
                selected = profiles.get("default", {}) if isinstance(profiles, dict) else {}
                for profile in profiles.values():
                    if not isinstance(profile, dict):
                        continue
                    match = profile.get("match") or {}
                    min_ram = match.get("min_ram_gb", 0)
                    max_ram = match.get("max_ram_gb", math.inf)
                    match_arch = match.get("arch")
                    if ram >= min_ram and ram <= max_ram and (match_arch is None or match_arch == arch):
                        selected = profile
                        break

                profile_defaults = selected.get("defaults") or {}
                if profile_defaults:
                    defaults = {
                        "vlm_model": profile_defaults.get("vlm_model", ""),
                        "text_model": profile_defaults.get("text_model", ""),
                        "required_models": profile_defaults.get("required_models", []),
                        "warmup_models": profile_defaults.get("warmup_models", []),
                    }
        except Exception:
            pass  # Fall through to safe defaults

    # Fallback to safe defaults (mac_32gb profile - works on more systems)
    if not defaults.get("vlm_model") or not defaults.get("text_model"):
        defaults = {
            "vlm_model": "qwen3-vl:8b-instruct-q4_K_M",
            "text_model": "qwen3:14b-q4_K_M",
            "required_models": ["qwen3-vl:8b-instruct-q4_K_M", "qwen3:14b-q4_K_M"],
            "warmup_models": ["qwen3-vl:8b-instruct-q4_K_M", "qwen3:14b-q4_K_M"],
        }

    return defaults


@functools.cache
def get_default_vlm_model() -> str:
    """Get the default VLM model name from configuration.

    Loads from environment variable, model profile, or safe fallback.

    Returns:
        Default VLM model name (e.g., "qwen3-vl:8b-instruct-q4_K_M").
    """
    defaults = _load_model_profile_defaults()
    return str(defaults.get("vlm_model", "qwen3-vl:8b-instruct-q4_K_M"))


@functools.cache
def get_default_text_model() -> str:
    """Get the default text model name from configuration.

    Loads from environment variable, model profile, or safe fallback.

    Returns:
        Default text model name (e.g., "qwen3:14b-q4_K_M").
    """
    defaults = _load_model_profile_defaults()
    return str(defaults.get("text_model", "qwen3:14b-q4_K_M"))


@functools.cache
def get_warmup_models() -> list[str]:
    """Get the list of models to pre-warm from configuration.

    Loads from environment variable, model profile, or safe fallback.

    Returns:
        List of model names to pre-warm.
    """
    defaults = _load_model_profile_defaults()
    warmup_models = defaults.get("warmup_models", [])
    if isinstance(warmup_models, list):
        return [str(model) for model in warmup_models]
    if isinstance(warmup_models, str):
        return [warmup_models]
    return []


@functools.cache
def get_allowed_models() -> set[str]:
    """Get the set of allowed models for the current hardware profile.

    Returns all models that are supported on this system based on the
    hardware profile configuration.

    Returns:
        Set of allowed model names (e.g., {"qwen3-vl:8b-instruct-q4_K_M", "qwen3:14b-q4_K_M"}).
    """
    defaults = _load_model_profile_defaults()
    required = defaults.get("required_models", [])
    warmup = defaults.get("warmup_models", [])
    required_set = {str(model) for model in required} if isinstance(required, list) else set()
    warmup_set = {str(model) for model in warmup} if isinstance(warmup, list) else set()
    # Combine required and warmup models, plus defaults
    allowed = required_set | warmup_set
    # Add default models if not already included
    if defaults.get("vlm_model"):
        allowed.add(str(defaults["vlm_model"]))
    if defaults.get("text_model"):
        allowed.add(str(defaults["text_model"]))
    return allowed


def is_model_allowed(model_name: str | None) -> bool:
    """Check if a model is allowed for the current hardware profile.

    Args:
        model_name: Model name to check. If None, returns True (will use default).

    Returns:
        True if model is allowed, False otherwise.
    """
    if model_name is None:
        return True  # None means use default, which is always allowed
    allowed = get_allowed_models()
    return model_name in allowed


__all__ = [
    "check_service_health",
    "ensure_service_running",
    "get_allowed_models",
    "get_client_path",
    "get_default_text_model",
    "get_default_vlm_model",
    "get_ollama_base_url",
    "get_project_root",
    "get_warmup_models",
    "import_client",
    "is_model_allowed",
]
