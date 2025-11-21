from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal Server Error: {e}"},
        )
