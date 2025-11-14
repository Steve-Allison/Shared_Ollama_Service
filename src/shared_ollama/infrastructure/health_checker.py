"""HTTP health checker for Ollama service.

This module provides infrastructure-level health checking functionality,
separating HTTP concerns from core business logic.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Literal

import requests


def check_ollama_health(
    base_url: str,
    timeout: int = 5,
) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
    """Check if the Ollama service is healthy via HTTP.

    Performs a lightweight health check by requesting the /api/tags endpoint.
    Returns a tuple indicating health status and optional error message.

    Args:
        base_url: Base URL for Ollama service (e.g., "http://localhost:11434").
        timeout: Request timeout in seconds. Defaults to 5 seconds.

    Returns:
        Tuple of (is_healthy, error_message):
            - (True, None) if service is healthy
            - (False, str) if service is unhealthy, with error message

    Side effects:
        Makes an HTTP GET request to the Ollama service.
    """
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


def check_ollama_health_simple(
    base_url: str,
    timeout: int = 5,
) -> bool:
    """Check if the Ollama service is healthy (simple boolean return).

    Simplified version that returns only a boolean. Useful for quick checks
    where error details are not needed.

    Args:
        base_url: Base URL for Ollama service.
        timeout: Request timeout in seconds.

    Returns:
        True if service responds with HTTP 200, False otherwise.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False

