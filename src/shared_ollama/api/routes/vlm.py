"""VLM routes for Vision-Language Model endpoints.

Provides both native Ollama format (/vlm) and OpenAI-compatible format (/vlm/openai)
for multimodal conversations with images and text.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from shared_ollama.api.dependencies import (
    get_request_context,
    get_vlm_queue,
    get_vlm_use_case,
)
from shared_ollama.api.mappers import (
    api_to_domain_vlm_request,
    api_to_domain_vlm_request_openai,
)
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import (
    ChatMessage,
    RequestContext,
    VLMRequest,
    VLMRequestOpenAI,
    VLMResponse,
)
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.core.queue import RequestQueue
from shared_ollama.domain.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

router = APIRouter()


def _map_http_status_code(status_code: int | None) -> tuple[int, str]:
    """Map Ollama HTTP status codes to appropriate API responses.

    Converts Ollama service HTTP status codes to appropriate FastAPI
    status codes and error messages for client consumption.

    Args:
        status_code: HTTP status code from Ollama service. None if
            status code unavailable.

    Returns:
        Tuple of (http_status_code, error_message):
            - (400, ...) for 4xx client errors from Ollama
            - (502, ...) for 5xx server errors from Ollama
            - (503, ...) for unknown/unavailable status
    """
    match status_code:
        case code if code and 400 <= code < 500:
            return (
                status.HTTP_400_BAD_REQUEST,
                f"Invalid request to Ollama service (status {code})",
            )
        case code if code and code >= 500:
            return (
                status.HTTP_502_BAD_GATEWAY,
                "Ollama service returned an error. Please try again later.",
            )
        case _:
            return (status.HTTP_503_SERVICE_UNAVAILABLE, "Ollama service is unavailable.")


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


@router.post("/vlm", tags=["VLM"])
@limiter.limit("30/minute")
async def vlm_chat(
    request: Request,
    use_case: VLMUseCase = Depends(get_vlm_use_case),
    queue: RequestQueue = Depends(get_vlm_queue),
) -> VLMResponse | StreamingResponse:
    """Vision-Language Model (VLM) chat completion endpoint.

    Processes multimodal conversations with images and text, returning the assistant's
    response. Supports image compression and caching for optimal performance.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            VLMRequest JSON with at least one image.
        use_case: VLMUseCase instance (injected via DI).
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

    # Parse request body
    try:
        body = await request.json()
        api_req = VLMRequest(**body)
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
        # Convert API model to domain entity (validation happens here)
        domain_req = api_to_domain_vlm_request(api_req)

        # Acquire VLM queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_vlm_requested: request_id=%s", ctx.request_id)
                # Use case returns AsyncIterator for streaming
                result = await use_case.execute(
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
            result = await use_case.execute(
                request=domain_req,
                request_id=ctx.request_id,
                client_ip=ctx.client_ip,
                project_name=ctx.project_name,
                stream=False,
                target_format=api_req.compression_format,
            )
            # Type narrowing: stream=False returns dict
            assert isinstance(result, dict)
            result_dict = result

            # Convert result dict to API response
            message_content = result_dict.get("message", {}).get("content", "")
            model_used = result_dict.get("model", "unknown")
            prompt_eval_count = result_dict.get("prompt_eval_count", 0)
            eval_count = result_dict.get("eval_count", 0)
            total_duration = result_dict.get("total_duration", 0)
            load_duration = result_dict.get("load_duration", 0)

            load_ms = load_duration / 1_000_000 if load_duration else 0.0
            total_ms = total_duration / 1_000_000 if total_duration else 0.0

            return VLMResponse(
                message=ChatMessage(role="assistant", content=message_content),
                model=model_used,
                request_id=ctx.request_id,
                latency_ms=0.0,  # Use case handles latency tracking internally
                model_load_ms=round(load_ms, 3) if load_ms else None,
                model_warm_start=load_ms == 0.0,
                images_processed=len(api_req.images),
                compression_savings_bytes=None,  # TODO: Get from use case result
                prompt_eval_count=prompt_eval_count,
                generation_eval_count=eval_count,
                total_duration_ms=round(total_ms, 3) if total_ms else None,
            )

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except (InvalidRequestError, ValueError) as exc:
        # ValueError from domain validation or InvalidRequestError from use case
        logger.warning("vlm_validation_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid VLM request: {exc!s}",
        ) from exc
    except ConnectionError as exc:
        logger.error("vlm_connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except TimeoutError as exc:
        logger.error("vlm_timeout_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="VLM request timed out. Large images may take longer to process.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else None
        http_status, error_msg = _map_http_status_code(status_code)
        logger.error(
            "vlm_http_status_error: request_id=%s, status_code=%s, error=%s",
            ctx.request_id,
            status_code,
            str(exc),
        )
        raise HTTPException(status_code=http_status, detail=error_msg) from exc
    except httpx.RequestError as exc:
        logger.error("vlm_request_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_vlm_completion: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        # Include request_id in error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


@router.post("/vlm/openai", tags=["VLM"])
@limiter.limit("30/minute")
async def vlm_chat_openai(
    request: Request,
    use_case: VLMUseCase = Depends(get_vlm_use_case),
    queue: RequestQueue = Depends(get_vlm_queue),
) -> VLMResponse | StreamingResponse:
    """OpenAI-compatible Vision-Language Model (VLM) chat completion endpoint.

    Processes multimodal conversations with OpenAI-compatible message format
    (images embedded in message content). For Docling and other OpenAI-compatible clients.
    Converted internally to native Ollama format for processing.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            VLMRequestOpenAI JSON with at least one image in message content.
        use_case: VLMUseCase instance (injected via DI).
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
        - Converts OpenAI format to native Ollama format
        - Validates at least one image present
        - Processes and compresses images (if enabled)
        - Checks image cache for duplicates
        - Acquires VLM queue slot (may wait or timeout)
        - Makes HTTP request to Ollama service
        - Logs request with VLM-specific metrics
        - Records metrics and analytics
    """
    ctx = get_request_context(request)

    # Parse request body (OpenAI-compatible format)
    try:
        body = await request.json()
        api_req = VLMRequestOpenAI(**body)
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

        # Count images for response metrics
        image_count = len(domain_req.images)

        # Acquire VLM queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if api_req.stream:
                logger.info("streaming_vlm_openai_requested: request_id=%s", ctx.request_id)
                # Use case returns AsyncIterator for streaming
                result = await use_case.execute(
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
            result = await use_case.execute(
                request=domain_req,
                request_id=ctx.request_id,
                client_ip=ctx.client_ip,
                project_name=ctx.project_name,
                stream=False,
                target_format=api_req.compression_format,
            )
            # Type narrowing: stream=False returns dict
            assert isinstance(result, dict)
            result_dict = result

            # Convert result dict to API response
            message_content = result_dict.get("message", {}).get("content", "")
            model_used = result_dict.get("model", "unknown")
            prompt_eval_count = result_dict.get("prompt_eval_count", 0)
            eval_count = result_dict.get("eval_count", 0)
            total_duration = result_dict.get("total_duration", 0)
            load_duration = result_dict.get("load_duration", 0)

            load_ms = load_duration / 1_000_000 if load_duration else 0.0
            total_ms = total_duration / 1_000_000 if total_duration else 0.0

            return VLMResponse(
                message=ChatMessage(role="assistant", content=message_content),
                model=model_used,
                request_id=ctx.request_id,
                latency_ms=0.0,  # Use case handles latency tracking internally
                model_load_ms=round(load_ms, 3) if load_ms else None,
                model_warm_start=load_ms == 0.0,
                images_processed=image_count,
                compression_savings_bytes=None,  # TODO: Get from use case result
                prompt_eval_count=prompt_eval_count,
                generation_eval_count=eval_count,
                total_duration_ms=round(total_ms, 3) if total_ms else None,
            )

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except (InvalidRequestError, ValueError) as exc:
        # ValueError from domain validation or InvalidRequestError from use case
        logger.warning("vlm_openai_validation_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid VLM request (OpenAI format): {exc!s}",
        ) from exc
    except ConnectionError as exc:
        logger.error("vlm_openai_connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except TimeoutError as exc:
        logger.error("vlm_openai_timeout_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="VLM request timed out. Large images may take longer to process.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else None
        http_status, error_msg = _map_http_status_code(status_code)
        logger.error(
            "vlm_openai_http_status_error: request_id=%s, status_code=%s, error=%s",
            ctx.request_id,
            status_code,
            str(exc),
        )
        raise HTTPException(status_code=http_status, detail=error_msg) from exc
    except httpx.RequestError as exc:
        logger.error("vlm_openai_request_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_vlm_openai_completion: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        # Include request_id in error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc
