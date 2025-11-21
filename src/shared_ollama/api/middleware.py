"""Middleware and error handlers for the API.

Provides rate limiting configuration, CORS middleware, and global
exception handlers for validation errors, rate limits, and unexpected errors.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from shared_ollama.api.dependencies import get_request_context
from shared_ollama.api.models import ErrorResponse
from shared_ollama.telemetry.structured_logging import log_request_event

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def _rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Temporary handler for rate limit exceeded during app setup."""
    retry_after = "60"
    if isinstance(exc, RateLimitExceeded):
        retry_after = str(getattr(exc, "retry_after", 60))
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "Rate limit exceeded"},
        headers={"Retry-After": retry_after},
    )


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that emits structured logs for every HTTP request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_time = time.perf_counter()
        status_code: int | None = None
        error_type: str | None = None
        error_message: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
            error_type = type(exc).__name__
            error_message = str(exc)
            raise
        finally:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
            try:
                ctx = get_request_context(request)
                event = {
                    "event": "http_request",
                    "request_id": ctx.request_id,
                    "client_ip": ctx.client_ip,
                    "project_name": ctx.project_name,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                }
            except Exception:
                event = {
                    "event": "http_request",
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                }
            if error_type:
                event["error_type"] = error_type
                event["error_message"] = error_message
            log_request_event(event)


def setup_middleware(app: FastAPI) -> None:
    """Configure middleware for the FastAPI application.

    Sets up:
    - Rate limiter (slowapi)
    - CORS middleware (allow all origins for development)

    Args:
        app: FastAPI application instance.
    """
    # Structured logging must run first to capture the full lifecycle
    app.add_middleware(StructuredLoggingMiddleware)

    # Add rate limiter to app
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers.

    Registers handlers for:
    - RequestValidationError (422)
    - RateLimitExceeded (429)
    - Exception (500 - catch-all)

    Args:
        app: FastAPI application instance.
    """

    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors.

        Global exception handler for Pydantic validation errors. Returns
        structured error response with validation details.

        Args:
            request: FastAPI Request object.
            exc: RequestValidationError with validation error details.

        Returns:
            JSONResponse with ErrorResponse containing validation error message
            and request_id. Status code 422 (Unprocessable Entity).
        """
        ctx = get_request_context(request)
        error_details = exc.errors()
        logger.warning("validation_error: request_id=%s, errors=%s", ctx.request_id, error_details)
        # Include full error details in response for debugging
        if error_details:
            first_error = error_details[0]
            error_msg = f"Validation error: {first_error.get('msg', 'Invalid request')} at {first_error.get('loc', [])}"
        else:
            error_msg = "Invalid request parameters"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error=error_msg,
                error_type="ValidationError",
                request_id=ctx.request_id,
            ).model_dump(),
        )

    async def rate_limit_exception_handler(
        request: Request, exc: RateLimitExceeded
    ) -> JSONResponse:
        """Handle rate limit exceeded errors.

        Global exception handler for rate limiting. Returns structured error
        response with Retry-After header.

        Args:
            request: FastAPI Request object.
            exc: RateLimitExceeded exception.

        Returns:
            JSONResponse with ErrorResponse. Status code 429 (Too Many Requests).
            Includes Retry-After header set to 60 seconds.
        """
        ctx = get_request_context(request)
        logger.warning(
            "rate_limit_exceeded: request_id=%s, client_ip=%s", ctx.request_id, ctx.client_ip
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=ErrorResponse(
                error="Rate limit exceeded. Please try again later.",
                error_type="RateLimitExceeded",
                request_id=ctx.request_id,
            ).model_dump(),
            headers={"Retry-After": "60"},
        )

    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unexpected errors.

        Catches any unhandled exceptions and returns a generic error response
        without exposing internal error details to clients.

        Args:
            request: FastAPI Request object.
            exc: Exception that was not handled by other handlers.

        Returns:
            JSONResponse with ErrorResponse containing generic error message.
            Status code 500 (Internal Server Error).

        Side effects:
            Logs full exception traceback for debugging.
        """
        ctx = get_request_context(request)
        logger.exception(
            "unhandled_exception: request_id=%s, error_type=%s, error=%s",
            ctx.request_id,
            type(exc).__name__,
            str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="An unexpected error occurred. Please try again later.",
                error_type=type(exc).__name__,
                request_id=ctx.request_id,
            ).model_dump(),
        )

    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.exception_handler(RateLimitExceeded)(rate_limit_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)
