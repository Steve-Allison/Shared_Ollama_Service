"""Utility helpers for the Shared Ollama Service.

This module provides common utilities for path resolution, service health checks,
and dynamic imports. All functions are designed to be stateless and cacheable
for performance.

Key behaviors:
    - Project root detection works for both editable installs and installed packages
    - Service health checks use timeouts and proper error handling
    - All path operations use pathlib for cross-platform compatibility
"""

from __future__ import annotations

import functools
import importlib
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import requests

if TYPE_CHECKING:
    from shared_ollama.client.sync import SharedOllamaClient


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

    match (package_root / "pyproject.toml").exists():
        case True:
            return package_root
        case False:
            # Fallback: walk up until we find repository marker
            for parent in Path(__file__).resolve().parents:
                if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                    return parent
            return package_root


def get_ollama_base_url() -> str:
    """Get the Ollama base URL from environment variables.

    Constructs the base URL from OLLAMA_BASE_URL, OLLAMA_HOST, and OLLAMA_PORT
    environment variables. Falls back to localhost:11434 if not set.

    Returns:
        Base URL string (e.g., "http://localhost:11434"). Always includes
        protocol and port, with trailing slashes removed.
    """
    import os

    if base_url := os.getenv("OLLAMA_BASE_URL"):
        return base_url.rstrip("/")

    host = os.getenv("OLLAMA_HOST", "localhost")
    port = os.getenv("OLLAMA_PORT", "11434")

    match ":" in host:
        case True:
            # Host already includes port
            return f"http://{host}"
        case False:
            return f"http://{host}:{port}"


def check_service_health(
    base_url: str | None = None,
    timeout: int = 5,
) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
    """Check if the Ollama service is healthy.

    Performs a lightweight health check by requesting the /api/tags endpoint.
    Returns a tuple indicating health status and optional error message.

    Args:
        base_url: Base URL for Ollama service. If None, uses get_ollama_base_url().
        timeout: Request timeout in seconds. Defaults to 5 seconds.

    Returns:
        Tuple of (is_healthy, error_message):
            - (True, None) if service is healthy
            - (False, str) if service is unhealthy, with error message

    Side effects:
        Makes an HTTP GET request to the Ollama service.
    """
    if base_url is None:
        base_url = get_ollama_base_url()

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        match response.status_code:
            case HTTPStatus.OK:
                return (True, None)
            case status_code:
                return (False, f"Service returned status code {status_code}")
    except requests.exceptions.ConnectionError:
        return (False, f"Cannot connect to {base_url}. Is the service running?")
    except requests.exceptions.Timeout:
        return (False, f"Connection to {base_url} timed out after {timeout}s")
    except Exception as exc:  # noqa: BLE001
        return (False, f"Unexpected error: {exc!s}")


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

    match (is_healthy, raise_on_fail):
        case (False, True):
            msg = (
                f"Ollama service is not available. {error}\n"
                "Start the service with: ./scripts/setup_launchd.sh\n"
                "Or manually: ollama serve"
            )
            raise ConnectionError(msg)
        case _:
            return is_healthy


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
    return getattr(module, "SharedOllamaClient")


__all__ = [
    "check_service_health",
    "ensure_service_running",
    "get_client_path",
    "get_ollama_base_url",
    "get_project_root",
    "import_client",
]
