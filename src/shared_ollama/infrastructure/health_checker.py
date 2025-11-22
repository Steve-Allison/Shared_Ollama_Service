"""HTTP health checker for Ollama service.

This module provides infrastructure-level health checking functionality,
separating HTTP concerns from core business logic. Health checks are used
for service discovery, startup validation, and monitoring.

Design Principles:
    - Infrastructure Layer: HTTP-specific code isolated from core logic
    - Lightweight: Uses /api/tags endpoint for fast health checks
    - Error Handling: Returns detailed error messages for debugging
    - Timeout Protection: Configurable timeouts prevent hanging checks

Health Check Strategy:
    - Endpoint: GET /api/tags (lightweight, doesn't load models)
    - Success: HTTP 200 response indicates healthy service
    - Failure: Non-200 status or connection errors indicate unhealthy
    - Timeout: Configurable timeout prevents indefinite waiting

Usage:
    Used by core utilities (check_service_health) and Ollama manager
    for service availability checks. Returns tuple format for detailed
    error reporting or simple boolean for quick checks.
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
    This endpoint is fast and doesn't require model loading, making it ideal
    for frequent health checks.

    Args:
        base_url: Base URL for Ollama service. Format: "http://host:port" or
            "https://host:port". Example: "http://localhost:11434".
        timeout: Request timeout in seconds. Must be positive. Default: 5.
            Health check fails if response takes longer than timeout.

    Returns:
        Tuple of (is_healthy, error_message):
            - (True, None) if service responds with HTTP 200
            - (False, str) if service is unhealthy, with descriptive error message:
                - Connection errors: "Cannot connect to {base_url}. Is the service running?"
                - Timeout errors: "Connection to {base_url} timed out after {timeout}s"
                - HTTP errors: "Service returned status code {code}"
                - Other errors: "Unexpected error: {exception}"

    Raises:
        No exceptions raised. All errors are caught and returned in tuple format.

    Side Effects:
        Makes an HTTP GET request to {base_url}/api/tags endpoint.

    Note:
        This function is designed for infrastructure-level health checking.
        It uses the requests library and handles all exceptions gracefully,
        returning error details in the tuple rather than raising exceptions.
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
    except Exception as exc:
        return (False, f"Unexpected error: {exc!s}")


def check_ollama_health_simple(
    base_url: str,
    timeout: int = 5,
) -> bool:
    """Check if the Ollama service is healthy (simple boolean return).

    Simplified version that returns only a boolean. Useful for quick checks
    where error details are not needed. Uses same health check strategy as
    check_ollama_health() but discards error messages.

    Args:
        base_url: Base URL for Ollama service. Format: "http://host:port" or
            "https://host:port". Example: "http://localhost:11434".
        timeout: Request timeout in seconds. Must be positive. Default: 5.

    Returns:
        True if service responds with HTTP 200, False for any other condition
        (connection errors, timeouts, non-200 status codes, exceptions).

    Note:
        This function is a convenience wrapper for cases where only a boolean
        result is needed. For detailed error information, use check_ollama_health().
        All exceptions are caught and return False.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code == HTTPStatus.OK
