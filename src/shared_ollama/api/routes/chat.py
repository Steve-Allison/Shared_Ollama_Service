"""Chat routes for text-only chat completion endpoint.

Provides the /chat endpoint for multi-turn text conversations. This endpoint
supports both streaming and non-streaming responses, with comprehensive error
handling and request queuing.

Key Features:
    - Multi-Turn Conversations: Chat with message history
    - Streaming Support: Server-Sent Events (SSE) for real-time responses
    - Request Queuing: Uses chat queue for concurrency control
    - Error Handling: Comprehensive error handling with structured logging
    - Rate Limiting: Integrated with slowapi rate limiting
    - Tool Calling: Supports OpenAI-compatible tool calling

Endpoint:
    POST /api/v1/chat
        - Request: ChatRequest (messages, model, options, stream, tools, etc.)
        - Response: ChatResponse (non-streaming) or SSE stream (streaming)
        - Rate Limited: Yes (via slowapi middleware)

Request Flow:
    1. Request validated by Pydantic (ChatRequest)
    2. Request queued via RequestQueue (concurrency control)
    3. Mapped to domain entity (ChatRequest)
    4. Executed via ChatUseCase
    5. Response built from use case result
    6. Returned as JSON or SSE stream

Streaming:
    - Set stream=True in request to enable SSE streaming
    - Stream format: "data: {json}\n\n" per chunk
    - Final chunk includes "done: true" marker

Tool Calling:
    - Supports OpenAI-compatible tool definitions and tool calls
    - Tools defined in request.tools
    - Tool calls returned in message.tool_calls
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
    get_chat_use_case,
    get_request_context,
)
from shared_ollama.api.error_handlers import handle_route_errors
from shared_ollama.api.mappers import api_to_domain_chat_request, api_to_domain_chat_request_openai
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import ChatRequest, ChatRequestOpenAI, RequestContext
from shared_ollama.api.response_builders import (
    build_chat_response,
    build_openai_chat_response,
    build_openai_stream_chunk,
    json_response,
)
from shared_ollama.api.type_guards import is_dict_result
from shared_ollama.application.use_cases import ChatUseCase
from shared_ollama.core.queue import RequestQueue

logger = logging.getLogger(__name__)

router = APIRouter()

UseCaseDep = Annotated[ChatUseCase, Depends(get_chat_use_case)]
QueueDep = Annotated[RequestQueue, Depends(get_chat_queue)]


async def _stream_chat_sse(
    stream_iter: AsyncIterator[dict[str, Any]],
    ctx: RequestContext,
) -> AsyncIterator[str]:
    """Stream chat responses in Server-Sent Events (SSE) format.

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
            "role": "assistant",
            "done": True,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "request_id": ctx.request_id,
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        logger.exception("Error during chat streaming: %s", exc)


@router.post("/chat", tags=["Chat"], response_model=None)
@limiter.limit("60/minute")
async def chat(
    request: Request,
    use_case: ChatUseCase = Depends(get_chat_use_case),  # noqa: B008
    queue: RequestQueue = Depends(get_chat_queue),  # noqa: B008
) -> Response:
    """Chat completion endpoint.

    Processes a conversation with multiple messages and returns the assistant's
    response. Supports both streaming (Server-Sent Events) and non-streaming
    responses based on the 'stream' parameter in the request body.

    Args:
        request: FastAPI Request object (injected). Body must contain
            ChatRequest JSON with messages list.
        use_case: ChatUseCase instance (injected via DI).
        queue: Chat request queue (injected via DI).

    Returns:
        - ChatResponse (JSON) if stream=False
        - StreamingResponse (text/event-stream) if stream=True

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Invalid messages or request parameters
            - 503: Ollama service unavailable
            - 504: Request timeout
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Validates message structure and content
        - Acquires queue slot (may wait or timeout)
        - Makes HTTP request to Ollama service
        - Logs request event with comprehensive metrics
        - Records metrics via MetricsCollector
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()
    api_req: ChatRequest | None = None
    model_name: str | None = None

    handle_error = handle_route_errors(
        ctx,
        "chat",
        start_time=start_time,
        event_builder=lambda: {
            "model": model_name,
            "stream": api_req.stream if api_req else None,
        },
    )

    # Parse request body
    try:
        body = await request.json()
        api_req = ChatRequest(**body)
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
        domain_req = api_to_domain_chat_request(api_req)
        domain_model = getattr(domain_req, "model", None)
        model_name = domain_model.value if domain_model else None

        # Acquire queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_chat_requested: request_id=%s", ctx.request_id)
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
                    _stream_chat_sse(result, ctx),
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

            response = build_chat_response(result, ctx)
            return json_response(response)

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except Exception as exc:
        handle_error(exc)  # This raises HTTPException


@router.post("/chat/completions", tags=["Chat"], response_model=None)
@limiter.limit("60/minute")
async def chat_completions(
    request: Request,
    use_case: ChatUseCase = Depends(get_chat_use_case),  # noqa: B008
    queue: RequestQueue = Depends(get_chat_queue),  # noqa: B008
) -> Response:
    """OpenAI-compatible chat completions endpoint.

    Processes a conversation with multiple messages and returns the assistant's
    response in OpenAI-compatible format. Supports both streaming (Server-Sent Events)
    and non-streaming responses based on the 'stream' parameter in the request body.

    This endpoint follows OpenAI's /chat/completions API specification for maximum
    compatibility with OpenAI clients and libraries.

    Args:
        request: FastAPI Request object (injected). Body must contain
            ChatRequestOpenAI JSON with messages list (OpenAI format).
        use_case: ChatUseCase instance (injected via DI).
        queue: Chat request queue (injected via DI).

    Returns:
        - OpenAI-compatible ChatCompletionResponse (JSON) if stream=False
        - StreamingResponse (text/event-stream) with OpenAI format if stream=True

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Invalid messages or request parameters
            - 503: Ollama service unavailable
            - 504: Request timeout
            - 500: Internal server error

    Side effects:
        - Parses request body JSON (OpenAI format)
        - Validates message structure and content
        - Acquires queue slot (may wait or timeout)
        - Makes HTTP request to Ollama service
        - Logs request event with comprehensive metrics
        - Records metrics via MetricsCollector
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()
    api_req: ChatRequestOpenAI | None = None
    model_name: str | None = None

    handle_error = handle_route_errors(
        ctx,
        "chat_completions",
        start_time=start_time,
        event_builder=lambda: {
            "model": model_name,
            "stream": api_req.stream if api_req else None,
        },
    )

    # Parse request body (OpenAI-compatible format)
    try:
        body = await request.json()
        api_req = ChatRequestOpenAI(**body)
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

        # Convert OpenAI-compatible API model to native Ollama domain entity
        domain_req = api_to_domain_chat_request_openai(api_req)
        domain_model = getattr(domain_req, "model", None)
        model_name = domain_model.value if domain_model else None

        # Acquire queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_chat_completions_requested: request_id=%s", ctx.request_id)
                # Use case returns AsyncIterator for streaming
                result_stream = await use_case.execute(
                    request=domain_req,
                    request_id=ctx.request_id,
                    client_ip=ctx.client_ip,
                    project_name=ctx.project_name,
                    stream=True,
                )
                assert not isinstance(result_stream, dict)

                created_ts = int(time.time())
                role_emitted = False

                async def openai_stream():
                    nonlocal role_emitted
                    async for chunk in result_stream:
                        openai_chunk = build_openai_stream_chunk(
                            chunk,
                            ctx,
                            created_ts=created_ts,
                            include_role=not role_emitted,
                        )
                        role_emitted = True
                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    # Send final [DONE] marker
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    openai_stream(),
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

            # Convert to OpenAI-compatible response format
            openai_response = build_openai_chat_response(result, ctx)
            return Response(
                content=json.dumps(openai_response),
                media_type="application/json",
            )

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except Exception as exc:
        handle_error(exc)  # This raises HTTPException
