import traceback
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response


async def catch_exceptions_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    try:
        return await call_next(request)
    except Exception as exc:  # pragma: no cover - debugging middleware
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal Server Error: {exc}"},
        )
