"""Shared error handling utilities for route handlers.

This module provides centralized error handling logic to reduce code duplication
across route handlers and ensure consistent error responses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NoReturn

from fastapi import HTTPException, status
import httpx

from shared_ollama.api.models import RequestContext
from shared_ollama.domain.exceptions import InvalidRequestError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def handle_route_errors(
    ctx: RequestContext,
    operation_name: str,
    timeout_message: str | None = None,
) -> Callable[[Exception], NoReturn]:
    """Create an error handler function for route handlers.

    Returns a function that handles common exceptions and converts them to
    appropriate HTTPException responses with consistent logging.

    Args:
        ctx: Request context with request_id for logging.
        operation_name: Name of the operation (e.g., "chat", "vlm") for logging.
        timeout_message: Custom timeout message. If None, uses default.

    Returns:
        Function that takes an exception and raises HTTPException.

    Example:
        >>> error_handler = handle_route_errors(ctx, "chat")
        >>> try:
        ...     result = await use_case.execute(...)
        ... except Exception as exc:
        ...     error_handler(exc)
    """
    default_timeout_msg = "Request timed out. The model may be taking longer than expected to respond."
    timeout_msg = timeout_message or default_timeout_msg

    def handle_error(exc: Exception) -> NoReturn:
        """Handle exception and raise appropriate HTTPException."""
        # Re-raise HTTPException as-is (guard clause)
        match exc:
            case HTTPException():
                raise

        # Use match/case for type-based exception handling (Python 3.13+ pattern)
        match exc:
            case InvalidRequestError() | ValueError():
                logger.warning(
                    f"{operation_name}_validation_error: request_id=%s, error=%s",
                    ctx.request_id,
                    str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid {operation_name} request: {exc!s}",
                ) from exc

            case ConnectionError():
                logger.error(
                    f"{operation_name}_connection_error: request_id=%s, error=%s",
                    ctx.request_id,
                    str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Ollama service is unavailable. Please check if the service is running.",
                ) from exc

            case TimeoutError():
                logger.error(
                    f"{operation_name}_timeout_error: request_id=%s, error=%s",
                    ctx.request_id,
                    str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=timeout_msg,
                ) from exc

            case httpx.HTTPStatusError() as http_exc:
                status_code = http_exc.response.status_code if http_exc.response else None
                http_status, error_msg = _map_http_status_code(status_code)
                logger.error(
                    f"{operation_name}_http_status_error: request_id=%s, status_code=%s, error=%s",
                    ctx.request_id,
                    status_code,
                    str(http_exc),
                )
                raise HTTPException(status_code=http_status, detail=error_msg) from http_exc

            case httpx.RequestError():
                logger.error(
                    f"{operation_name}_request_error: request_id=%s, error=%s",
                    ctx.request_id,
                    str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to connect to Ollama service. Please check if the service is running.",
                ) from exc

            case _:
                # Handle all other exceptions
                logger.exception(
                    f"unexpected_error_{operation_name}: request_id=%s, error_type=%s",
                    ctx.request_id,
                    type(exc).__name__,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
                ) from exc

    return handle_error


def _map_http_status_code(status_code: int | None) -> tuple[int, str]:
    """Map HTTP status code to FastAPI status and error message.

    Uses match/case for cleaner pattern matching (Python 3.13+).

    Args:
        status_code: HTTP status code from Ollama response.

    Returns:
        Tuple of (FastAPI status code, error message).
    """
    match status_code:
        case 400:
            return status.HTTP_400_BAD_REQUEST, "Invalid request to Ollama service."
        case 404:
            return status.HTTP_404_NOT_FOUND, "Model not found in Ollama service."
        case 429:
            return status.HTTP_503_SERVICE_UNAVAILABLE, "Ollama service is rate limiting requests."
        case 500:
            return status.HTTP_503_SERVICE_UNAVAILABLE, "Ollama service encountered an internal error."
        case 503:
            return status.HTTP_503_SERVICE_UNAVAILABLE, "Ollama service is temporarily unavailable."
        case _:
            # Default for unknown status codes
            return status.HTTP_503_SERVICE_UNAVAILABLE, "Ollama service returned an error."

