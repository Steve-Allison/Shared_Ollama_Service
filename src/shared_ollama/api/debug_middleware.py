"""Debug middleware for catching and logging exceptions.

This module provides a simple debug middleware that catches all exceptions
and returns them as JSON responses with traceback information. Useful for
development and debugging, but should be disabled in production.

Warning:
    This middleware catches ALL exceptions, including those that should be
    handled by proper error handlers. It's intended for debugging only and
    should not be used in production environments.

Usage:
    Add this middleware to FastAPI app for development debugging:
        app.middleware("http")(catch_exceptions_middleware)
"""

import traceback
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response


async def catch_exceptions_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Catch all exceptions and return JSON response with traceback.

    This middleware catches all exceptions that occur during request processing
    and returns a JSON response with the error message and traceback. Useful
    for debugging during development.

    Args:
        request: FastAPI request object.
        call_next: Callable that processes the request and returns response.

    Returns:
        Response object. If exception occurs, returns JSONResponse with
        status 500 and error message. Otherwise returns normal response.

    Side Effects:
        - Prints traceback to stderr for debugging
        - Returns JSON error response instead of raising exception

    Note:
        This middleware should only be used in development. In production,
        use proper error handlers (error_handlers.py) for consistent error
        responses and security.
    """
    try:
        return await call_next(request)
    except Exception as exc:  # pragma: no cover - debugging middleware
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal Server Error: {exc}"},
        )
