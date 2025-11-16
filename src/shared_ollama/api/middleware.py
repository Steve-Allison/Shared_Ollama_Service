"""Middleware and error handlers for the API.

Provides rate limiting configuration, CORS middleware, and global
exception handlers for validation errors, rate limits, and unexpected errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shared_ollama.api.dependencies import get_request_context
from shared_ollama.api.models import ErrorResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Temporary handler for rate limit exceeded during app setup."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "Rate limit exceeded"},
        headers={"Retry-After": "60"},
    )


def setup_middleware(app: FastAPI) -> None:
    """Configure middleware for the FastAPI application.

    Sets up:
    - Rate limiter (slowapi)
    - CORS middleware (allow all origins for development)

    Args:
        app: FastAPI application instance.
    """
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

    @app.exception_handler(RequestValidationError)
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

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
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

    @app.exception_handler(Exception)
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
