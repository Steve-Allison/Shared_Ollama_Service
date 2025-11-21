"""Generation routes for text generation endpoint.

Provides the /generate endpoint for single-prompt text generation. This endpoint
supports both streaming and non-streaming responses, with comprehensive error
handling and request queuing.

Key Features:
    - Single-Prompt Generation: Generate text from a single prompt
    - Streaming Support: Server-Sent Events (SSE) for real-time text generation
    - Request Queuing: Uses chat queue for concurrency control
    - Error Handling: Comprehensive error handling with structured logging
    - Rate Limiting: Integrated with slowapi rate limiting

Endpoint:
    POST /api/v1/generate
        - Request: GenerateRequest (prompt, model, options, stream, etc.)
        - Response: GenerateResponse (non-streaming) or SSE stream (streaming)
        - Rate Limited: Yes (via slowapi middleware)

Request Flow:
    1. Request validated by Pydantic (GenerateRequest)
    2. Request queued via RequestQueue (concurrency control)
    3. Mapped to domain entity (GenerationRequest)
    4. Executed via GenerateUseCase
    5. Response built from use case result
    6. Returned as JSON or SSE stream

Streaming:
    - Set stream=True in request to enable SSE streaming
    - Stream format: "data: {json}\n\n" per chunk
    - Final chunk includes "done: true" marker
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from shared_ollama.api.dependencies import (
    get_chat_queue,
    get_generate_use_case,
    get_request_context,
)
from shared_ollama.api.error_handlers import handle_route_errors
from shared_ollama.api.mappers import api_to_domain_generation_request
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import GenerateRequest, RequestContext
from shared_ollama.api.response_builders import build_generate_response, json_response
from shared_ollama.api.type_guards import is_dict_result
from shared_ollama.application.use_cases import GenerateUseCase
from shared_ollama.core.queue import RequestQueue

logger = logging.getLogger(__name__)

router = APIRouter()

UseCaseDep = Annotated[GenerateUseCase, Depends(get_generate_use_case)]
QueueDep = Annotated[RequestQueue, Depends(get_chat_queue)]


async def _stream_generate_sse(
    stream_iter: AsyncIterator[dict[str, Any]],
    ctx: RequestContext,
) -> AsyncIterator[str]:
    """Stream generate responses in Server-Sent Events (SSE) format.

    Converts async generator chunks from use case into SSE-formatted strings.

    Args:
        stream_iter: AsyncIterator from use case.
        ctx: Request context for tracking.

    Yields:
        SSE-formatted strings. Each chunk is prefixed with "data: " and
        suffixed with "\\n\\n". Final chunk on error includes error details.
    """
    try:
        async for chunk_data in stream_iter:
            chunk_json = json.dumps(chunk_data)
            yield f"data: {chunk_json}\n\n"
    except Exception as exc:
        error_chunk = {
            "chunk": "",
            "done": True,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "request_id": ctx.request_id,
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        logger.exception("Error during generate streaming: %s", exc)


@router.post("/generate", tags=["Generation"], response_model=None)
@limiter.limit("60/minute")
async def generate(
    request: Request,
    use_case: GenerateUseCase = Depends(get_generate_use_case),  # noqa: B008
    queue: RequestQueue = Depends(get_chat_queue),  # noqa: B008
) -> Response:
    """Generate text from a prompt.

    Sends a text generation request to the Ollama service. Supports both
    streaming (Server-Sent Events) and non-streaming responses based on
    the 'stream' parameter in the request body.

    Args:
        request: FastAPI Request object (injected). Body must contain
            GenerateRequest JSON.
        use_case: GenerateUseCase instance (injected via DI).
        queue: Chat request queue (injected via DI).

    Returns:
        - GenerateResponse (JSON) if stream=False
        - StreamingResponse (text/event-stream) if stream=True

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Invalid prompt or request parameters
            - 503: Ollama service unavailable
            - 504: Request timeout
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Acquires queue slot (may wait or timeout)
        - Makes HTTP request to Ollama service
        - Logs request event with comprehensive metrics
        - Records metrics via MetricsCollector
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()
    api_req: GenerateRequest | None = None
    model_name: str | None = None

    handle_error = handle_route_errors(
        ctx,
        "generate",
        start_time=start_time,
        event_builder=lambda: {
            "model": model_name,
            "stream": api_req.stream if api_req else None,
        },
    )

    # Parse request body
    try:
        body = await request.json()
        api_req = GenerateRequest(**body)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in request body: {e!s}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e!s}",
        ) from e

    try:
        # Validate model is allowed for current hardware profile
        from shared_ollama.core.utils import get_allowed_models, is_model_allowed

        requested_model = api_req.model
        if requested_model and not is_model_allowed(requested_model):
            allowed = get_allowed_models()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model '{requested_model}' is not supported on this hardware profile. "
                    f"Allowed models: {', '.join(sorted(allowed))}"
                ),
            )

        # Convert API model to domain entity (validation happens here)
        domain_req = api_to_domain_generation_request(api_req)
        domain_model = getattr(domain_req, "model", None)
        model_name = domain_model.value if domain_model else None

        # Acquire queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_generate_requested: request_id=%s", ctx.request_id)
                # Use case returns AsyncIterator for streaming
                result = await use_case.execute(
                    request=domain_req,
                    request_id=ctx.request_id,
                    client_ip=ctx.client_ip,
                    project_name=ctx.project_name,
                    stream=True,
                )
                # Type narrowing: stream=True returns AsyncIterator
                assert not isinstance(result, dict)
                return StreamingResponse(
                    _stream_generate_sse(result, ctx),
                    media_type="text/event-stream",
                )

            # Non-streaming: use case handles all business logic, logging, and metrics
            result = await use_case.execute(
                request=domain_req,
                request_id=ctx.request_id,
                client_ip=ctx.client_ip,
                project_name=ctx.project_name,
                stream=False,
            )
            # Type narrowing: stream=False returns dict
            if not is_dict_result(result):
                raise RuntimeError("Expected dict result for non-streaming request")

            response = build_generate_response(result, ctx)
            return json_response(response)

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except Exception as exc:
        handle_error(exc)  # This raises HTTPException
