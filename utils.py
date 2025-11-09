"""
Utility functions and helpers for Shared Ollama Service.
Provides service discovery, configuration, and helper functions.
"""

import os
import sys
from http import HTTPStatus
from pathlib import Path

import requests


def get_ollama_base_url() -> str:
    """
    Get Ollama base URL from environment or use default.

    Checks in order:
    1. OLLAMA_BASE_URL environment variable
    2. OLLAMA_HOST environment variable (constructs URL)
    3. Default: http://localhost:11434

    Returns:
        Base URL string
    """
    # Check explicit base URL first
    base_url = os.getenv("OLLAMA_BASE_URL")
    if base_url:
        return base_url.rstrip("/")

    # Check for host/port separately
    host = os.getenv("OLLAMA_HOST", "localhost")
    port = os.getenv("OLLAMA_PORT", "11434")

    # Handle different host formats
    if ":" in host:
        # Already has port
        return f"http://{host}"
    return f"http://{host}:{port}"


def check_service_health(base_url: str | None = None, timeout: int = 5) -> tuple[bool, str | None]:
    """
    Check if Ollama service is healthy and accessible.

    Args:
        base_url: Optional base URL (uses get_ollama_base_url() if not provided)
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_healthy, error_message)
        is_healthy: True if service is accessible
        error_message: Error description if unhealthy, None if healthy
    """
    if base_url is None:
        base_url = get_ollama_base_url()

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if response.status_code == HTTPStatus.OK:
            return True, None
        return False, f"Service returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {base_url}. Is the service running?"
    except requests.exceptions.Timeout:
        return False, f"Connection to {base_url} timed out after {timeout}s"
    except Exception as e:
        return False, f"Unexpected error: {e!s}"


def ensure_service_running(base_url: str | None = None, raise_on_fail: bool = True) -> bool:
    """
    Ensure Ollama service is running, optionally raise exception if not.

    Args:
        base_url: Optional base URL
        raise_on_fail: If True, raises ConnectionError if service is not running

    Returns:
        True if service is running

    Raises:
        ConnectionError: If service is not running and raise_on_fail is True
    """
    is_healthy, error = check_service_health(base_url)

    if not is_healthy and raise_on_fail:
        msg = (
            f"Ollama service is not available. {error}\n"
            "Start the service with: ./scripts/setup_launchd.sh\n"
            "Or manually: ollama serve"
        )
        raise ConnectionError(msg)

    return is_healthy


def get_project_root() -> Path | None:
    """
    Get the Shared_Ollama_Service project root directory.

    This is useful for projects that need to find the service directory
    to import the client or access configuration.

    Returns:
        Path to project root, or None if not found
    """
    # Try common locations
    current = Path(__file__).resolve().parent

    # If this file is in the project, return parent
    if current.name == "Shared_Ollama_Service" or (current / "shared_ollama_client.py").exists():
        return current

    # Look for project in common locations
    common_paths = [
        Path.home() / "AI_Projects+Code" / "Shared_Ollama_Service",
        Path.home() / "Projects" / "Shared_Ollama_Service",
        Path("/opt") / "Shared_Ollama_Service",
    ]

    for path in common_paths:
        if path.exists() and (path / "shared_ollama_client.py").exists():
            return path

    # Check if SHARED_OLLAMA_SERVICE_PATH is set
    env_path = os.getenv("SHARED_OLLAMA_SERVICE_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    return None


def get_client_path() -> Path | None:
    """
    Get path to shared_ollama_client.py for importing.

    Returns:
        Path to shared_ollama_client.py, or None if not found
    """
    project_root = get_project_root()
    if project_root:
        client_path = project_root / "shared_ollama_client.py"
        if client_path.exists():
            return client_path
    return None


def import_client() -> type:
    """
    Dynamically import and return the SharedOllamaClient class.

    This allows projects to use the client without modifying sys.path.

    Returns:
        SharedOllamaClient class

    Raises:
        ImportError: If client cannot be found or imported
    """
    client_path = get_client_path()
    if not client_path:
        msg = (
            "Cannot find shared_ollama_client.py. "
            "Set SHARED_OLLAMA_SERVICE_PATH environment variable to the project root, "
            "or ensure the file is in your Python path."
        )
        raise ImportError(msg)

    # Add project root to path temporarily
    project_root = client_path.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from shared_ollama_client import SharedOllamaClient  # noqa: PLC0415
    except ImportError as e:
        msg = f"Failed to import SharedOllamaClient: {e}"
        raise ImportError(msg) from e
    else:
        return SharedOllamaClient
