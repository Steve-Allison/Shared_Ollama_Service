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
    - Request queue for concurrency control
    - Helper functions for error handling and status code mapping

Endpoints:
    - GET /api/v1/health - Health check
    - GET /api/v1/models - List available models
    - GET /api/v1/queue/stats - Queue statistics
    - POST /api/v1/generate - Text generation (with streaming support)
    - POST /api/v1/chat - Chat completion (with streaming support)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shared_ollama.api.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateStreamChunk,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    QueueStatsResponse,
    RequestContext,
)
from shared_ollama.client import (
    AsyncOllamaConfig,
    AsyncSharedOllamaClient,
    GenerateOptions,
)
from shared_ollama.core.ollama_manager import initialize_ollama_manager
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import check_service_health, get_project_root
from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.structured_logging import log_request_event

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """Manage application lifespan."""
    global _client, _queue

    # Debug: Log that lifespan is starting
    print("LIFESPAN: Starting Shared Ollama Service API", flush=True)
    logger.info("LIFESPAN: Starting Shared Ollama Service API")

    # Initialize and start Ollama manager (manages Ollama process internally)
    logger.info("LIFESPAN: Initializing Ollama manager")
    try:
        project_root = get_project_root()
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        ollama_manager = initialize_ollama_manager(
            base_url="http://localhost:11434",
            log_dir=log_dir,
            auto_detect_optimizations=True,
        )

        logger.info("LIFESPAN: Starting Ollama service (managed internally)")
        ollama_started = await ollama_manager.start(wait_for_ready=True, max_wait_time=30)
        if not ollama_started:
            logger.error("LIFESPAN: Failed to start Ollama service")
            raise RuntimeError("Failed to start Ollama service. Check logs for details.")
        logger.info("LIFESPAN: Ollama service started successfully")
        print("LIFESPAN: Ollama service started", flush=True)
    except Exception as exc:
        logger.error("LIFESPAN: Failed to start Ollama service: %s", exc, exc_info=True)
        print(f"LIFESPAN ERROR: Failed to start Ollama: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # Initialize async client (connects to the managed Ollama service)
    try:
        config = AsyncOllamaConfig()
        logger.info("LIFESPAN: Creating AsyncSharedOllamaClient")
        # Don't verify on init - we'll do it manually to ensure it completes
        _client = AsyncSharedOllamaClient(config=config, verify_on_init=False)
        logger.info("LIFESPAN: Client created, ensuring initialization")
        # Ensure client is initialized and verified (async)
        await _client._ensure_client()
        logger.info("LIFESPAN: Client ensured, verifying connection")
        await _client._verify_connection()
        logger.info("LIFESPAN: Ollama async client initialized successfully")
        print("LIFESPAN: Client initialized successfully", flush=True)
    except Exception as exc:
        logger.error("LIFESPAN: Failed to initialize Ollama async client: %s", exc, exc_info=True)
        print(f"LIFESPAN ERROR: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        _client = None
        # Don't raise - allow server to start but client will be None
        # This way we can see the error in logs

    # Initialize request queue
    logger.info("LIFESPAN: Initializing RequestQueue")
    _queue = RequestQueue(max_concurrent=3, max_queue_size=50, default_timeout=60.0)
    logger.info("LIFESPAN: RequestQueue initialized (max_concurrent=3, max_queue_size=50)")
    print("LIFESPAN: RequestQueue initialized", flush=True)

    yield

    # Shutdown
    logger.info("Shutting down Shared Ollama Service API")
    if _client:
        try:
            # Close the async client properly (we're in an async context)
            await _client.close()
        except Exception as exc:
            logger.warning("Error closing async client: %s", exc)
        finally:
            _client = None

    _queue = None

    # Stop Ollama service (managed internally)
    try:
        from shared_ollama.core.ollama_manager import get_ollama_manager

        logger.info("LIFESPAN: Stopping Ollama service")
        ollama_manager = get_ollama_manager()
        await ollama_manager.stop(timeout=10)
        logger.info("LIFESPAN: Ollama service stopped")
        print("LIFESPAN: Ollama service stopped", flush=True)
    except Exception as exc:
        logger.warning("Error stopping Ollama service: %s", exc)


app = FastAPI(
    title="Shared Ollama Service API",
    description="RESTful API for the Shared Ollama Service - Language-agnostic access to Ollama models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan_context,
)

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

# Global async client instance and request queue
_client: AsyncSharedOllamaClient | None = None
_queue: RequestQueue | None = None


def get_client() -> AsyncSharedOllamaClient:
    """Get the global async client instance."""
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama async client not initialized",
        )
    return _client


def get_queue() -> RequestQueue:
    """Get the global request queue instance."""
    if _queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Request queue not initialized",
        )
    return _queue


def get_request_context(request: Request) -> RequestContext:
    """Extract request context from FastAPI request.

    Creates a RequestContext object with unique request ID and extracted
    headers. Used throughout request lifecycle for logging and tracking.

    Args:
        request: FastAPI Request object.

    Returns:
        RequestContext with request_id, client_ip, user_agent, and project_name.
    """
    return RequestContext(
        request_id=str(uuid.uuid4()),
        client_ip=get_remote_address(request),
        user_agent=request.headers.get("user-agent"),
        project_name=request.headers.get("x-project-name"),
    )


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
            return (status.HTTP_400_BAD_REQUEST, f"Invalid request to Ollama service (status {code})")
        case code if code and code >= 500:
            return (status.HTTP_502_BAD_GATEWAY, "Ollama service returned an error. Please try again later.")
        case _:
            return (status.HTTP_503_SERVICE_UNAVAILABLE, "Ollama service is unavailable.")


def _log_and_record_error(
    ctx: RequestContext,
    operation: str,
    model: str,
    latency_ms: float,
    error_type: str,
    error_message: str,
    status_code: int | None = None,
    **extra_log_data: dict[str, Any],
) -> None:
    """Log error and record metrics with consistent format.

    Helper function to reduce code duplication across error handlers.

    Args:
        ctx: Request context.
        operation: Operation name (e.g., "generate", "chat").
        model: Model name or "unknown".
        latency_ms: Request latency in milliseconds.
        error_type: Type of error.
        error_message: Error message.
        status_code: Optional HTTP status code.
        **extra_log_data: Additional data to include in logs.
    """
    log_data: dict[str, Any] = {
        "event": "api_request",
        "client_type": "rest_api",
        "operation": operation,
        "status": "error",
        "model": model,
        "request_id": ctx.request_id,
        "client_ip": ctx.client_ip,
        "project_name": ctx.project_name,
        "latency_ms": round(latency_ms, 3),
        "error_type": error_type,
        "error_message": error_message,
    }
    if status_code is not None:
        log_data["ollama_status_code"] = status_code
    log_data.update(extra_log_data)

    log_request_event(log_data)

    error_name = f"{error_type}:{status_code}" if status_code else error_type
    MetricsCollector.record_request(
        model=model,
        operation=operation,
        latency_ms=latency_ms,
        success=False,
        error=error_name,
    )


async def _stream_generate(
    client: AsyncSharedOllamaClient,
    ctx: RequestContext,
    generate_req: GenerateRequest,
    options: GenerateOptions | None,
) -> AsyncIterator[str]:
    """Stream generate responses in Server-Sent Events (SSE) format.

    Converts async generator chunks from client.generate_stream() into
    SSE-formatted strings. Handles errors by sending error chunk as final
    message.

    Args:
        client: AsyncSharedOllamaClient instance.
        ctx: Request context for tracking.
        generate_req: GenerateRequest with prompt and options.
        options: GenerateOptions for generation parameters.

    Yields:
        SSE-formatted strings. Each chunk is prefixed with "data: " and
        suffixed with "\\n\\n". Final chunk on error includes error details.

    Side effects:
        - Calls client.generate_stream() which makes HTTP request
        - Logs exceptions on error
    """
    try:
        async for chunk_data in client.generate_stream(
            prompt=generate_req.prompt,
            model=generate_req.model,
            system=generate_req.system,
            options=options,
        ):
            # Convert chunk to SSE format
            chunk_json = json.dumps(chunk_data)
            yield f"data: {chunk_json}\n\n"

    except Exception as exc:
        # Send error as final chunk
        error_chunk = {
            "chunk": "",
            "done": True,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "request_id": ctx.request_id,
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        logger.exception("Error during generate streaming: %s", exc)


async def _stream_chat(
    client: AsyncSharedOllamaClient,
    ctx: RequestContext,
    chat_req: ChatRequest,
    messages: list[dict[str, str]],
    options: GenerateOptions | None,
) -> AsyncIterator[str]:
    """Stream chat responses in Server-Sent Events (SSE) format.

    Converts async generator chunks from client.chat_stream() into
    SSE-formatted strings. Handles errors by sending error chunk as final
    message.

    Args:
        client: AsyncSharedOllamaClient instance.
        ctx: Request context for tracking.
        chat_req: ChatRequest with messages and options.
        messages: List of message dictionaries for client.
        options: GenerateOptions for generation parameters.

    Yields:
        SSE-formatted strings. Each chunk is prefixed with "data: " and
        suffixed with "\\n\\n". Final chunk on error includes error details.

    Side effects:
        - Calls client.chat_stream() which makes HTTP request
        - Logs exceptions on error
    """
    try:
        async for chunk_data in client.chat_stream(
            messages=messages,
            model=chat_req.model,
            options=options,
        ):
            # Convert chunk to SSE format
            chunk_json = json.dumps(chunk_data)
            yield f"data: {chunk_json}\n\n"

    except Exception as exc:
        # Send error as final chunk
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


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Performs a lightweight health check on the API and underlying Ollama
    service. Returns status information without requiring authentication.

    Returns:
        HealthResponse with:
            - status: "healthy" or "unhealthy"
            - ollama_service: Ollama service status string
            - version: API version string

    Side effects:
        Calls check_service_health() which makes HTTP request to Ollama.
    """
    is_healthy, error = check_service_health()
    ollama_status = "healthy" if is_healthy else f"unhealthy: {error}"

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        ollama_service=ollama_status,
        version="1.0.0",
    )


@app.get("/api/v1/queue/stats", response_model=QueueStatsResponse, tags=["Queue"])
async def get_queue_stats() -> QueueStatsResponse:
    """Get request queue statistics.

    Returns comprehensive queue metrics including current state, historical
    counts, and performance statistics.

    Returns:
        QueueStatsResponse with:
            - Current state: queued, in_progress
            - Historical counts: completed, failed, rejected, timeout
            - Performance metrics: wait times (total, max, avg)
            - Configuration: max_concurrent, max_queue_size, default_timeout

    Side effects:
        Acquires asyncio.Lock briefly to read statistics atomically.
    """
    queue = get_queue()
    stats = await queue.get_stats()
    config = queue.get_config()

    return QueueStatsResponse(
        queued=stats.queued,
        in_progress=stats.in_progress,
        completed=stats.completed,
        failed=stats.failed,
        rejected=stats.rejected,
        timeout=stats.timeout,
        total_wait_time_ms=stats.total_wait_time_ms,
        max_wait_time_ms=stats.max_wait_time_ms,
        avg_wait_time_ms=stats.avg_wait_time_ms,
        max_concurrent=config["max_concurrent"],
        max_queue_size=config["max_queue_size"],
        default_timeout=config["default_timeout"],
    )


@app.get("/api/v1/models", response_model=ModelsResponse, tags=["Models"])
@limiter.limit("30/minute")
async def list_models(request: Request) -> ModelsResponse:
    """List available models.

    Retrieves the list of all models available in the Ollama service.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected).

    Returns:
        ModelsResponse with list of ModelInfo objects, one per available model.

    Raises:
        HTTPException: If Ollama service is unavailable or request fails.

    Side effects:
        - Makes HTTP request to Ollama service
        - Logs request event
        - Records metrics
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()

    try:
        client = get_client()
        models_data = await client.list_models()

        latency_ms = (time.perf_counter() - start_time) * 1000

        models = [
            ModelInfo(
                name=model.get("name", "unknown"),
                size=model.get("size"),
                modified_at=model.get("modified_at"),
            )
            for model in models_data
        ]

        # Log request
        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "list_models",
            "status": "success",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
        })

        MetricsCollector.record_request(
            model="system",
            operation="list_models",
            latency_ms=latency_ms,
            success=True,
        )

        return ModelsResponse(models=models)

    except ConnectionError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "list_models",
            "status": "error",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "ConnectionError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model="system",
            operation="list_models",
            latency_ms=latency_ms,
            success=False,
            error="ConnectionError",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        status_code = exc.response.status_code if exc.response else None

        if status_code and 400 <= status_code < 500:
            http_status = status.HTTP_400_BAD_REQUEST
            error_msg = f"Invalid request to Ollama service (status {status_code})"
        elif status_code and status_code >= 500:
            http_status = status.HTTP_502_BAD_GATEWAY
            error_msg = "Ollama service returned an error. Please try again later."
        else:
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
            error_msg = "Ollama service is unavailable."

        logger.error("http_status_error: request_id=%s, status_code=%s, error=%s", ctx.request_id, status_code, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "list_models",
            "status": "error",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "HTTPStatusError",
            "error_message": str(exc),
            "ollama_status_code": status_code,
        })

        MetricsCollector.record_request(
            model="system",
            operation="list_models",
            latency_ms=latency_ms,
            success=False,
            error=f"HTTPStatusError:{status_code}",
        )

        raise HTTPException(
            status_code=http_status,
            detail=error_msg,
        ) from exc
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("request_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "list_models",
            "status": "error",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "RequestError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model="system",
            operation="list_models",
            latency_ms=latency_ms,
            success=False,
            error="RequestError",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("unexpected_error_listing_models: request_id=%s, error_type=%s", ctx.request_id, type(exc).__name__)

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "list_models",
            "status": "error",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model="system",
            operation="list_models",
            latency_ms=latency_ms,
            success=False,
            error=type(exc).__name__,
        )

        # Don't expose internal error details - log them but return generic message
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later or contact support.",
        ) from exc


@app.post("/api/v1/generate", tags=["Generation"])
@limiter.limit("60/minute")
async def generate(request: Request) -> Response:
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
    start_time = time.perf_counter()

    # Manually parse request body to avoid slowapi decorator interference
    try:
        body = await request.json()
        generate_req = GenerateRequest(**body)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in request body: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {str(e)}",
        ) from e

    try:
        client = get_client()
        queue = get_queue()

        # Validate prompt
        if not generate_req.prompt or not generate_req.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate prompt length (reasonable limit)
        if len(generate_req.prompt) > 1_000_000:  # 1M characters
            raise ValueError("Prompt is too long. Maximum length is 1,000,000 characters")

        # Build options
        options = None
        if any([
            generate_req.temperature is not None,
            generate_req.top_p is not None,
            generate_req.top_k is not None,
            generate_req.max_tokens is not None,
            generate_req.seed is not None,
            generate_req.stop is not None,
        ]):
            options = GenerateOptions(
                temperature=generate_req.temperature,
                top_p=generate_req.top_p,
                top_k=generate_req.top_k,
                max_tokens=generate_req.max_tokens,
                seed=generate_req.seed,
                stop=generate_req.stop,
            )

        # Acquire queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if generate_req.stream:
                logger.info("streaming_generate_requested: request_id=%s", ctx.request_id)
                return StreamingResponse(
                    _stream_generate(client, ctx, generate_req, options),
                    media_type="text/event-stream",
                )

            # Non-streaming generate (async)
        result = await client.generate(
            prompt=generate_req.prompt,
            model=generate_req.model,
            system=generate_req.system,
            options=options,
            stream=False,
            format=generate_req.format,  # Pass format parameter to Ollama
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        load_ms = result.load_duration / 1_000_000 if result.load_duration else 0.0
        total_ms = result.total_duration / 1_000_000 if result.total_duration else 0.0

        # Log request
        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "generate",
            "status": "success",
            "model": result.model,
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "total_duration_ms": round(total_ms, 3) if total_ms else None,
            "model_load_ms": round(load_ms, 3) if load_ms else 0.0,
            "model_warm_start": load_ms == 0.0,
            "prompt_chars": len(generate_req.prompt),
            "prompt_eval_count": result.prompt_eval_count,
            "generation_eval_count": result.eval_count,
            "stream": generate_req.stream,
        })

        return GenerateResponse(
            text=result.text,
            model=result.model,
            request_id=ctx.request_id,
            latency_ms=round(latency_ms, 3),
            model_load_ms=round(load_ms, 3) if load_ms else None,
            model_warm_start=load_ms == 0.0,
            prompt_eval_count=result.prompt_eval_count,
            generation_eval_count=result.eval_count,
            total_duration_ms=round(total_ms, 3) if total_ms else None,
        )

    except HTTPException:
        # Re-raise HTTPException (from parsing or other validation) - FastAPI will handle it
        raise
    except ValueError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning("validation_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "generate",
            "status": "error",
            "model": generate_req.model or "unknown" if 'generate_req' in locals() else "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "ValueError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=generate_req.model or "unknown",
            operation="generate",
            latency_ms=latency_ms,
            success=False,
            error="ValueError",
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {exc!s}",
        ) from exc
    except ConnectionError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "generate",
            "status": "error",
            "model": generate_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "ConnectionError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=generate_req.model or "unknown",
            operation="generate",
            latency_ms=latency_ms,
            success=False,
            error="ConnectionError",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except TimeoutError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("timeout_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "generate",
            "status": "error",
            "model": generate_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "TimeoutError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=generate_req.model or "unknown",
            operation="generate",
            latency_ms=latency_ms,
            success=False,
            error="TimeoutError",
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. The model may be taking longer than expected to respond.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        status_code = exc.response.status_code if exc.response else None

        # Map HTTP status codes using helper function
        http_status, error_msg = _map_http_status_code(status_code)

        logger.error("http_status_error: request_id=%s, status_code=%s, error=%s", ctx.request_id, status_code, str(exc))

        _log_and_record_error(
            ctx=ctx,
            operation="generate",
            model=generate_req.model or "unknown",
            latency_ms=latency_ms,
            error_type="HTTPStatusError",
            error_message=str(exc),
            status_code=status_code,
        )

        raise HTTPException(
            status_code=http_status,
            detail=error_msg,
        ) from exc
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("request_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "generate",
            "status": "error",
            "model": generate_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "RequestError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=generate_req.model or "unknown",
            operation="generate",
            latency_ms=latency_ms,
            success=False,
            error="RequestError",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("unexpected_error_generating_text: request_id=%s, error_type=%s", ctx.request_id, type(exc).__name__)

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "generate",
            "status": "error",
            "model": generate_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=generate_req.model or "unknown",
            operation="generate",
            latency_ms=latency_ms,
            success=False,
            error=type(exc).__name__,
        )

        # Don't expose internal error details - log them but return generic message
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later or contact support.",
        ) from exc


@app.post("/api/v1/chat", tags=["Chat"])
@limiter.limit("60/minute")
async def chat(request: Request) -> Response:
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
    start_time = time.perf_counter()

    try:
        # Manually parse request body to avoid slowapi decorator interference
        try:
            body = await request.json()
            chat_req = ChatRequest(**body)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid JSON in request body: {str(e)}",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Request validation failed: {str(e)}",
            ) from e

        client = get_client()
        queue = get_queue()

        # Validate messages
        if not chat_req.messages:
            raise ValueError("Messages list cannot be empty")

        # Validate message structure and content
        for i, msg in enumerate(chat_req.messages):
            if not msg.role or msg.role not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role '{msg.role}' in message {i}. Must be 'user', 'assistant', or 'system'")
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message {i} content cannot be empty")

        # Validate total message length
        total_length = sum(len(msg.content) for msg in chat_req.messages)
        if total_length > 1_000_000:  # 1M characters
            raise ValueError("Total message content is too long. Maximum length is 1,000,000 characters")

        # Convert messages to format expected by client
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_req.messages]

        # Build options
        options = None
        if any([
            chat_req.temperature is not None,
            chat_req.top_p is not None,
            chat_req.top_k is not None,
            chat_req.max_tokens is not None,
            chat_req.seed is not None,
            chat_req.stop is not None,
        ]):
            options = GenerateOptions(
                temperature=chat_req.temperature,
                top_p=chat_req.top_p,
                top_k=chat_req.top_k,
                max_tokens=chat_req.max_tokens,
                seed=chat_req.seed,
                stop=chat_req.stop,
            )

        # Acquire queue slot for request processing
        async with queue.acquire(request_id=ctx.request_id):
            # Handle streaming if requested
            if chat_req.stream:
                logger.info("streaming_chat_requested: request_id=%s", ctx.request_id)
                return StreamingResponse(
                    _stream_chat(client, ctx, chat_req, messages, options),
                    media_type="text/event-stream",
                )

            # Non-streaming chat (async)
        result = await client.chat(
                messages=messages, model=chat_req.model, options=options, stream=False
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract metrics from result (chat returns dict with message and metadata)
        message_content = result.get("message", {}).get("content", "")
        model_used = result.get("model", chat_req.model or "unknown")
        prompt_eval_count = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)
        total_duration = result.get("total_duration", 0)
        load_duration = result.get("load_duration", 0)

        load_ms = load_duration / 1_000_000 if load_duration else 0.0
        total_ms = total_duration / 1_000_000 if total_duration else 0.0

        # Log request
        total_prompt_chars = sum(len(msg.content) for msg in chat_req.messages)
        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "success",
            "model": model_used,
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "total_duration_ms": round(total_ms, 3) if total_ms else None,
            "model_load_ms": round(load_ms, 3) if load_ms else 0.0,
            "model_warm_start": load_ms == 0.0,
            "prompt_chars": total_prompt_chars,
            "prompt_eval_count": prompt_eval_count,
            "generation_eval_count": eval_count,
            "stream": chat_req.stream,
        })

        return ChatResponse(
            message=ChatMessage(role="assistant", content=message_content),
            model=model_used,
            request_id=ctx.request_id,
            latency_ms=round(latency_ms, 3),
            model_load_ms=round(load_ms, 3) if load_ms else None,
            model_warm_start=load_ms == 0.0,
            prompt_eval_count=prompt_eval_count,
            generation_eval_count=eval_count,
            total_duration_ms=round(total_ms, 3) if total_ms else None,
        )

    except ValueError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning("validation_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "error",
            "model": chat_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "ValueError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=chat_req.model or "unknown",
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error="ValueError",
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {exc!s}",
        ) from exc
    except ConnectionError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "error",
            "model": chat_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "ConnectionError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=chat_req.model or "unknown",
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error="ConnectionError",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except TimeoutError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("timeout_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "error",
            "model": chat_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "TimeoutError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=chat_req.model or "unknown",
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error="TimeoutError",
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. The model may be taking longer than expected to respond.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        status_code = exc.response.status_code if exc.response else None

        # Map HTTP status codes appropriately
        if status_code and 400 <= status_code < 500:
            http_status = status.HTTP_400_BAD_REQUEST
            error_msg = f"Invalid request to Ollama service (status {status_code})"
        elif status_code and status_code >= 500:
            http_status = status.HTTP_502_BAD_GATEWAY
            error_msg = "Ollama service returned an error. Please try again later."
        else:
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE
            error_msg = "Ollama service is unavailable."

        logger.error("http_status_error: request_id=%s, status_code=%s, error=%s", ctx.request_id, status_code, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "error",
            "model": chat_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "HTTPStatusError",
            "error_message": str(exc),
            "ollama_status_code": status_code,
        })

        MetricsCollector.record_request(
            model=chat_req.model or "unknown",
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error=f"HTTPStatusError:{status_code}",
        )

        raise HTTPException(
            status_code=http_status,
            detail=error_msg,
        ) from exc
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("request_error: request_id=%s, error=%s", ctx.request_id, str(exc))

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "error",
            "model": chat_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": "RequestError",
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=chat_req.model or "unknown",
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error="RequestError",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("unexpected_error_chat_completion: request_id=%s, error_type=%s", ctx.request_id, type(exc).__name__)

        log_request_event({
            "event": "api_request",
            "client_type": "rest_api",
            "operation": "chat",
            "status": "error",
            "model": chat_req.model or "unknown",
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "project_name": ctx.project_name,
            "latency_ms": round(latency_ms, 3),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })

        MetricsCollector.record_request(
            model=chat_req.model or "unknown",
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error=type(exc).__name__,
        )

        # Don't expose internal error details - log them but return generic message
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later or contact support.",
        ) from exc


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
    logger.warning("rate_limit_exceeded: request_id=%s, client_ip=%s", ctx.request_id, ctx.client_ip)
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
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
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
    logger.exception("unhandled_exception: request_id=%s, error_type=%s, error=%s", ctx.request_id, type(exc).__name__, str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An unexpected error occurred. Please try again later.",
            error_type=type(exc).__name__,
            request_id=ctx.request_id,
        ).model_dump(),
    )


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
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }
