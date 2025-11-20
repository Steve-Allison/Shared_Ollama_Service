"""VLM routes for Vision-Language Model endpoints.

Provides both native Ollama format (/vlm) and OpenAI-compatible format (/vlm/openai)
for multimodal conversations with images and text.
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
    get_request_context,
    get_vlm_queue,
    get_vlm_use_case,
)
from shared_ollama.api.error_handlers import handle_route_errors
from shared_ollama.api.mappers import (
    api_to_domain_vlm_request,
    api_to_domain_vlm_request_openai,
)
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import (
    RequestContext,
    VLMRequest,
    VLMRequestOpenAI,
)
from shared_ollama.api.response_builders import (
    build_openai_chat_response,
    build_openai_stream_chunk,
    build_vlm_response,
    json_response,
)
from shared_ollama.api.type_guards import is_dict_result
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.core.queue import RequestQueue

logger = logging.getLogger(__name__)

router = APIRouter()


async def _stream_chat_sse(
    stream_iter: AsyncIterator[dict[str, Any]],
    ctx: RequestContext,
) -> AsyncIterator[str]:
    """Stream chat responses in Server-Sent Events (SSE) format.

    Converts async generator chunks from use case into SSE-formatted strings.
    Used by both native and OpenAI-compatible VLM endpoints.

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
        logger.exception("Error during VLM streaming: %s", exc)


@router.post("/vlm", tags=["VLM"], response_model=None)
@limiter.limit("30/minute")
async def vlm_chat(
    request: Request,
    vlm_use_case_dep: Annotated[VLMUseCase, Depends(get_vlm_use_case)],
    queue: Annotated[RequestQueue, Depends(get_vlm_queue)],
) -> Response:
    """Vision-Language Model (VLM) chat completion endpoint.

    Processes multimodal conversations with images and text, returning the assistant's
    response. Supports image compression and caching for optimal performance.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            VLMRequest JSON with at least one image.
        vlm_use_case_dep: VLMUseCase instance (injected via DI).
        queue: VLM request queue (injected via DI).

    Returns:
        - VLMResponse (JSON) if stream=False
        - StreamingResponse (text/event-stream) if stream=True

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Invalid messages, missing images, or invalid image format
            - 503: Ollama service unavailable
            - 504: Request timeout (120s for VLM)
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Validates at least one image present
        - Processes and compresses images (if enabled)
        - Checks image cache for duplicates
        - Acquires VLM queue slot (may wait or timeout)
        - Makes HTTP request to Ollama service
        - Logs request with VLM-specific metrics
        - Records metrics and analytics
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()
    api_req: VLMRequest | None = None
    event_data: dict[str, Any] = {
        "model": None,
        "stream": None,
        "images_count": None,
        "compression_format": None,
        "image_compression": None,
        "max_dimension": None,
    }

    handle_error = handle_route_errors(
        ctx,
        "vlm",
        timeout_message="VLM request timed out. Large images may take longer to process.",
        start_time=start_time,
        event_builder=lambda: {k: v for k, v in event_data.items() if v is not None},
    )

    # Parse request body
    try:
        body = await request.json()
        api_req = VLMRequest(**body)
        event_data["stream"] = api_req.stream
        event_data["compression_format"] = api_req.compression_format
        event_data["image_compression"] = api_req.image_compression
        event_data["max_dimension"] = api_req.max_dimension
        event_data["images_count"] = len(api_req.images)
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
        domain_req = api_to_domain_vlm_request(api_req)
        event_data["model"] = (
            domain_req.model.value if getattr(domain_req, "model", None) else None
        )

        # Acquire VLM queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_vlm_requested: request_id=%s", ctx.request_id)
                # Use case returns AsyncIterator for streaming
                result = await vlm_use_case_dep.execute(
                    request=domain_req,
                    request_id=ctx.request_id,
                    client_ip=ctx.client_ip,
                    project_name=ctx.project_name,
                    stream=True,
                    target_format=api_req.compression_format,
                )
                # Type narrowing: stream=True returns AsyncIterator
                assert not isinstance(result, dict)
                return StreamingResponse(
                    _stream_chat_sse(result, ctx),
                    media_type="text/event-stream",
                )

            # Non-streaming: use case handles all business logic, logging, and metrics
            result = await vlm_use_case_dep.execute(
                request=domain_req,
                request_id=ctx.request_id,
                client_ip=ctx.client_ip,
                project_name=ctx.project_name,
                stream=False,
                target_format=api_req.compression_format,
            )
            # Type narrowing: stream=False returns dict
            if not is_dict_result(result):
                raise RuntimeError("Expected dict result for non-streaming request")

            response = build_vlm_response(result, ctx, images_processed=len(api_req.images))
            return json_response(response)

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except Exception as exc:
        handle_error(exc)  # This raises HTTPException


@router.post("/vlm/openai", tags=["VLM"], response_model=None)
@limiter.limit("30/minute")
async def vlm_chat_openai(
    request: Request,
    vlm_use_case_dep: Annotated[VLMUseCase, Depends(get_vlm_use_case)],
    queue: Annotated[RequestQueue, Depends(get_vlm_queue)],
) -> Response:
    """OpenAI-compatible Vision-Language Model (VLM) chat completion endpoint.

    Processes multimodal conversations with OpenAI-compatible message format
    (images embedded in message content). For Docling and other OpenAI-compatible clients.
    Converted internally to native Ollama format for processing.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            VLMRequestOpenAI JSON with at least one image in message content.
        vlm_use_case_dep: VLMUseCase instance (injected via DI).
        queue: VLM request queue (injected via DI).

    Returns:
        - VLMResponse (JSON) if stream=False
        - StreamingResponse (text/event-stream) if stream=True
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()
    api_req: VLMRequestOpenAI | None = None
    event_data: dict[str, Any] = {
        "model": None,
        "stream": None,
        "images_count": None,
        "compression_format": None,
    }

    handle_error = handle_route_errors(
        ctx,
        "vlm_openai",
        timeout_message="VLM request timed out. Large images may take longer to process.",
        start_time=start_time,
        event_builder=lambda: {k: v for k, v in event_data.items() if v is not None},
    )

    # Parse request body (OpenAI-compatible format)
    try:
        body = await request.json()
        api_req = VLMRequestOpenAI(**body)
        event_data["stream"] = api_req.stream
        event_data["compression_format"] = api_req.compression_format
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
        # Convert OpenAI-compatible API model to native Ollama domain entity
        domain_req = api_to_domain_vlm_request_openai(api_req)
        event_data["model"] = (
            domain_req.model.value if getattr(domain_req, "model", None) else None
        )

        # Count images for response metrics
        image_count = len(domain_req.images)
        event_data["images_count"] = image_count

        # Acquire VLM queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_vlm_openai_requested: request_id=%s", ctx.request_id)
                # Use case returns AsyncIterator for streaming
                result_stream = await vlm_use_case_dep.execute(
                    request=domain_req,
                    request_id=ctx.request_id,
                    client_ip=ctx.client_ip,
                    project_name=ctx.project_name,
                    stream=True,
                    target_format=api_req.compression_format,
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

                return StreamingResponse(
                    openai_stream(),
                    media_type="text/event-stream",
                )

            result = await vlm_use_case_dep.execute(
                request=domain_req,
                request_id=ctx.request_id,
                client_ip=ctx.client_ip,
                project_name=ctx.project_name,
                stream=False,
                target_format=api_req.compression_format,
            )
            if not is_dict_result(result):
                raise RuntimeError("Expected dict result for non-streaming request")

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
