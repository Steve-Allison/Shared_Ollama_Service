"\"\"\"Utility helpers for the Shared Ollama Service.\"\"\""

from __future__ import annotations

import importlib
import os
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:  # pragma: no cover
    from shared_ollama.client.sync import SharedOllamaClient


def get_ollama_base_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL")
    if base_url:
        return base_url.rstrip("/")

    host = os.getenv("OLLAMA_HOST", "localhost")
    port = os.getenv("OLLAMA_PORT", "11434")

    if ":" in host:
        return f"http://{host}"
    return f"http://{host}:{port}"


def check_service_health(base_url: str | None = None, timeout: int = 5) -> tuple[bool, str | None]:
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
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc!s}"


def ensure_service_running(base_url: str | None = None, raise_on_fail: bool = True) -> bool:
    is_healthy, error = check_service_health(base_url)

    if not is_healthy and raise_on_fail:
        msg = (
            f"Ollama service is not available. {error}\n"
            "Start the service with: ./scripts/setup_launchd.sh\n"
            "Or manually: ollama serve"
        )
        raise ConnectionError(msg)

    return is_healthy


def get_project_root() -> Path:
    """
    Return the project root directory.

    Works for both editable installs (src layout) and installed packages.
    """

    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "pyproject.toml").exists():
        return package_root

    # Fallback: walk up until we find repository marker
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return package_root


def get_client_path() -> Path:
    """
    Return the path to the synchronous client module.
    """

    return (get_project_root() / "src" / "shared_ollama" / "client" / "sync.py").resolve()


def import_client() -> type["SharedOllamaClient"]:
    """
    Dynamically import and return the `SharedOllamaClient` class.
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

