"""FastAPI REST API server for the Shared Ollama Service.

This module provides a language-agnostic REST API that wraps the Python
client library, enabling centralized logging, metrics, and control for
all projects.

Key behaviors:
    - Manages Ollama service lifecycle internally via OllamaManager
    - Implements request queuing for graceful traffic handling
    - Provides rate limiting via slowapi
    - Comprehensive error handling with consistent error responses
    - Streaming support via Server-Sent Events (SSE)
    - Automatic metrics collection and structured logging

Architecture:
    - FastAPI application with lifespan management
    - Global async client instance for Ollama operations
    - Separate request queues for chat (6 concurrent) and VLM (3 concurrent)
    - Image processing infrastructure with compression and caching
    - Helper functions for error handling and status code mapping

Endpoints:
    - GET /api/v1/health - Health check
    - GET /api/v1/models - List available models
    - GET /api/v1/queue/stats - Chat queue statistics
    - GET /api/v1/metrics - Service metrics
    - GET /api/v1/performance/stats - Performance statistics
    - GET /api/v1/analytics - Analytics report
    - POST /api/v1/generate - Text generation (with streaming support)
    - POST /api/v1/chat - Text-only chat completion (with streaming support)
    - POST /api/v1/vlm - Vision-Language Model chat (with image support)
    - POST /api/v1/batch/chat - Batch text-only chat processing (max 50 requests)
    - POST /api/v1/batch/vlm - Batch VLM processing (max 20 requests)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse

# Import modular components
from shared_ollama.api.lifespan import lifespan_context
from shared_ollama.api.middleware import limiter, setup_exception_handlers, setup_middleware
from shared_ollama.api.routes import system_router

from shared_ollama.api.dependencies import (
    get_batch_chat_use_case,
    get_batch_vlm_use_case,
    get_chat_queue,
    get_chat_use_case,
    get_generate_use_case,
    get_list_models_use_case,
    get_vlm_queue,
    get_vlm_use_case,
    set_dependencies,
)
from shared_ollama.api.mappers import (
    api_to_domain_chat_request,
    api_to_domain_generation_request,
    api_to_domain_vlm_request,
    api_to_domain_vlm_request_openai,
    domain_to_api_model_info,
)
from shared_ollama.api.models import (
    BatchChatRequest,
    BatchResponse,
    BatchVLMRequest,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelsResponse,
    QueueStatsResponse,
    RequestContext,
    VLMMessage,
    VLMRequest,
    VLMRequestOpenAI,
    VLMResponse,
)
from shared_ollama.application.batch_use_cases import (
    BatchChatUseCase,
    BatchVLMUseCase,
)
from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient
from shared_ollama.core.ollama_manager import initialize_ollama_manager
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import check_service_health, get_project_root
from shared_ollama.domain.exceptions import InvalidRequestError
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)
from shared_ollama.infrastructure.image_cache import ImageCache
from shared_ollama.infrastructure.image_processing import ImageProcessor

logger = logging.getLogger(__name__)

# Create FastAPI app with modular lifespan
app = FastAPI(
    title="Shared Ollama Service API",
    description="RESTful API for the Shared Ollama Service - Unified text and VLM endpoints with batch support",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan_context,
)

# Setup middleware and exception handlers (modular)
setup_middleware(app)
setup_exception_handlers(app)

# Include modular routers
app.include_router(system_router, prefix="/api/v1")


# Request context extraction (moved to dependencies.py, kept here for backward compatibility)
def get_request_context(request: Request) -> RequestContext:
    """Extract request context from FastAPI request.

    Creates a RequestContext object with unique request ID and extracted
    headers. Used throughout request lifecycle for logging and tracking.

    Args:
        request: FastAPI Request object.

    Returns:
        RequestContext with request_id, client_ip, user_agent, and project_name.
    """
    from shared_ollama.api.dependencies import get_request_context as _get_request_context

    return _get_request_context(request)


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


# Removed _log_and_record_error - use cases now handle all logging and metrics


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


# System endpoints (health, models, queue stats, metrics, analytics) are now in routes/system.py


@app.post("/api/v1/generate", tags=["Generation"])
@limiter.limit("60/minute")
async def generate(
    request: Request,
    use_case: GenerateUseCase = Depends(get_generate_use_case),
    queue: RequestQueue = Depends(get_chat_queue),
) -> Response:
    """Generate text from a prompt.

    Sends a text generation request to the Ollama service. Supports both
    streaming (Server-Sent Events) and non-streaming responses based on
    the 'stream' parameter in the request body.

    Args:
        request: FastAPI Request object (injected). Body must contain
            GenerateRequest JSON.

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
        # Convert API model to domain entity (validation happens here)
        domain_req = api_to_domain_generation_request(api_req)

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
            assert isinstance(result, dict)
            result_dict = result

            # Convert result dict to API response
            load_ms = (
                result_dict.get("load_duration", 0) / 1_000_000
                if result_dict.get("load_duration")
                else 0.0
            )
            total_ms = (
                result_dict.get("total_duration", 0) / 1_000_000
                if result_dict.get("total_duration")
                else 0.0
            )

            return GenerateResponse(
                text=result_dict.get("text", ""),
                model=result_dict.get("model", "unknown"),
                request_id=ctx.request_id,
                latency_ms=0.0,  # Use case handles latency tracking internally
                model_load_ms=round(load_ms, 3) if load_ms else None,
                model_warm_start=load_ms == 0.0,
                prompt_eval_count=result_dict.get("prompt_eval_count", 0),
                generation_eval_count=result_dict.get("eval_count", 0),
                total_duration_ms=round(total_ms, 3) if total_ms else None,
            )

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except (InvalidRequestError, ValueError) as exc:
        # ValueError from domain validation or InvalidRequestError from use case
        logger.warning("validation_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {exc!s}",
        ) from exc
    except ConnectionError as exc:
        logger.error("connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except TimeoutError as exc:
        logger.error("timeout_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. The model may be taking longer than expected to respond.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else None
        http_status, error_msg = _map_http_status_code(status_code)
        logger.error(
            "http_status_error: request_id=%s, status_code=%s, error=%s",
            ctx.request_id,
            status_code,
            str(exc),
        )
        raise HTTPException(status_code=http_status, detail=error_msg) from exc
    except httpx.RequestError as exc:
        logger.error("request_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_generating_text: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        # Include request_id in error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


@app.post("/api/v1/chat", tags=["Chat"])
@limiter.limit("60/minute")
async def chat(
    request: Request,
    use_case: ChatUseCase = Depends(get_chat_use_case),
    queue: RequestQueue = Depends(get_chat_queue),
) -> Response:
    """Chat completion endpoint.

    Processes a conversation with multiple messages and returns the assistant's
    response. Supports both streaming (Server-Sent Events) and non-streaming
    responses based on the 'stream' parameter in the request body.

    Args:
        request: FastAPI Request object (injected). Body must contain
            ChatRequest JSON with messages list.

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
        # Convert API model to domain entity (validation happens here)
        domain_req = api_to_domain_chat_request(api_req)

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

            return ChatResponse(
                message=ChatMessage(role="assistant", content=message_content),
                model=model_used,
                request_id=ctx.request_id,
                latency_ms=0.0,  # Use case handles latency tracking internally
                model_load_ms=round(load_ms, 3) if load_ms else None,
                model_warm_start=load_ms == 0.0,
                prompt_eval_count=prompt_eval_count,
                generation_eval_count=eval_count,
                total_duration_ms=round(total_ms, 3) if total_ms else None,
            )

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation)
        raise
    except (InvalidRequestError, ValueError) as exc:
        # ValueError from domain validation or InvalidRequestError from use case
        logger.warning("validation_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {exc!s}",
        ) from exc
    except ConnectionError as exc:
        logger.error("connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except TimeoutError as exc:
        logger.error("timeout_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. The model may be taking longer than expected to respond.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else None
        http_status, error_msg = _map_http_status_code(status_code)
        logger.error(
            "http_status_error: request_id=%s, status_code=%s, error=%s",
            ctx.request_id,
            status_code,
            str(exc),
        )
        raise HTTPException(status_code=http_status, detail=error_msg) from exc
    except httpx.RequestError as exc:
        logger.error("request_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_chat_completion: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        # Include request_id in error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


@app.post("/api/v1/vlm", tags=["VLM"])
@limiter.limit("30/minute")
async def vlm_chat(
    request: Request,
    use_case: VLMUseCase = Depends(get_vlm_use_case),
    queue: RequestQueue = Depends(get_vlm_queue),
) -> Response:
    """Vision-Language Model (VLM) chat completion endpoint.

    Processes multimodal conversations with images and text, returning the assistant's
    response. Supports image compression and caching for optimal performance.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            VLMRequest JSON with at least one image.
        use_case: VLMUseCase instance (injected via dependency injection).
        queue: VLM request queue (injected via dependency injection).

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


@app.post("/api/v1/vlm/openai", tags=["VLM"])
@limiter.limit("30/minute")
async def vlm_chat_openai(
    request: Request,
    use_case: VLMUseCase = Depends(get_vlm_use_case),
    queue: RequestQueue = Depends(get_vlm_queue),
) -> Response:
    """OpenAI-compatible Vision-Language Model (VLM) chat completion endpoint.

    Processes multimodal conversations with OpenAI-compatible message format
    (images embedded in message content). For Docling and other OpenAI-compatible clients.
    Converted internally to native Ollama format for processing.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            VLMRequestOpenAI JSON with at least one image in message content.
        use_case: VLMUseCase instance (injected via dependency injection).
        queue: VLM request queue (injected via dependency injection).

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


@app.post("/api/v1/batch/chat", tags=["Batch"])
@limiter.limit("10/minute")
async def batch_chat(
    request: Request,
    use_case: BatchChatUseCase = Depends(get_batch_chat_use_case),
) -> BatchResponse:
    """Batch text-only chat completion endpoint.

    Processes multiple chat requests in parallel, returning all results in a single
    response. Rate limited to 10 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            BatchChatRequest JSON with list of chat requests (max 50).
        use_case: BatchChatUseCase instance (injected via dependency injection).

    Returns:
        BatchResponse with:
            - batch_id: Unique batch identifier
            - total_requests: Number of requests in batch
            - successful: Count of successful requests
            - failed: Count of failed requests
            - total_time_ms: Total batch processing time
            - results: List of individual results with success/error status

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Too many requests in batch (>50)
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Processes up to 5 requests concurrently
        - Logs all individual requests
        - Records batch-level metrics
    """
    ctx = get_request_context(request)

    # Parse request body
    try:
        body = await request.json()
        api_req = BatchChatRequest(**body)
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
        # Convert API requests to domain entities
        domain_requests = [api_to_domain_chat_request(req) for req in api_req.requests]

        # Execute batch
        logger.info(
            "batch_chat_requested: request_id=%s, count=%d", ctx.request_id, len(domain_requests)
        )
        result = await use_case.execute(
            requests=domain_requests,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
        )

        return BatchResponse(**result)

    except HTTPException:
        # Re-raise HTTPException
        raise
    except (InvalidRequestError, ValueError) as exc:
        logger.warning(
            "batch_chat_validation_error: request_id=%s, error=%s", ctx.request_id, str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid batch chat request: {exc!s}",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_batch_chat: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


@app.post("/api/v1/batch/vlm", tags=["Batch"])
@limiter.limit("5/minute")
async def batch_vlm(
    request: Request,
    use_case: BatchVLMUseCase = Depends(get_batch_vlm_use_case),
) -> BatchResponse:
    """Batch VLM chat completion endpoint.

    Processes multiple VLM requests in parallel, with image compression and caching.
    Rate limited to 5 requests per minute per IP address due to resource intensity.

    Args:
        request: FastAPI Request object (injected). Body must contain
            BatchVLMRequest JSON with list of VLM requests (max 20).
        use_case: BatchVLMUseCase instance (injected via dependency injection).

    Returns:
        BatchResponse with:
            - batch_id: Unique batch identifier
            - total_requests: Number of requests in batch
            - successful: Count of successful requests
            - failed: Count of failed requests
            - total_time_ms: Total batch processing time
            - results: List of individual results with success/error status

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Too many requests in batch (>20) or invalid images
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Processes up to 3 VLM requests concurrently (resource-intensive)
        - Compresses and caches images
        - Logs all individual requests with VLM metrics
        - Records batch-level metrics
    """
    ctx = get_request_context(request)

    # Parse request body
    try:
        body = await request.json()
        api_req = BatchVLMRequest(**body)
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
        # Convert API requests to domain entities
        domain_requests = [api_to_domain_vlm_request(req) for req in api_req.requests]

        # Execute batch
        logger.info(
            "batch_vlm_requested: request_id=%s, count=%d", ctx.request_id, len(domain_requests)
        )
        result = await use_case.execute(
            requests=domain_requests,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
            target_format=api_req.compression_format,
        )

        return BatchResponse(**result)

    except HTTPException:
        # Re-raise HTTPException
        raise
    except (InvalidRequestError, ValueError) as exc:
        logger.warning(
            "batch_vlm_validation_error: request_id=%s, error=%s", ctx.request_id, str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid batch VLM request: {exc!s}",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_batch_vlm: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint with API information.

    Provides basic API metadata and links to documentation endpoints.
    Useful for service discovery and health checks.

    Returns:
        Dictionary with keys:
            - service: Service name
            - version: API version
            - docs: Path to OpenAPI documentation
            - health: Path to health check endpoint
    """
    return {
        "service": "Shared Ollama Service API",
        "version": "2.0.0",
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }
