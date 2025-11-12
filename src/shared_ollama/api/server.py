"""
FastAPI REST API server for the Shared Ollama Service.

Provides a language-agnostic REST API that wraps the Python client library,
enabling centralized logging, metrics, and control for all projects.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shared_ollama.api.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    RequestContext,
)
from shared_ollama.client import (
    AsyncOllamaConfig,
    AsyncSharedOllamaClient,
    GenerateOptions,
)
from shared_ollama.core.utils import check_service_health
from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.structured_logging import log_request_event

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """Manage application lifespan."""
    global _client

    # Startup
    logger.info("Starting Shared Ollama Service API")
    try:
        config = AsyncOllamaConfig()
        _client = AsyncSharedOllamaClient(config=config, verify_on_init=True)
        # Ensure client is initialized (async)
        await _client._ensure_client()
        logger.info("Ollama async client initialized successfully")
    except Exception as exc:
        logger.error("Failed to initialize Ollama async client: %s", exc)
        raise

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

# Global async client instance
_client: AsyncSharedOllamaClient | None = None


def get_client() -> AsyncSharedOllamaClient:
    """Get the global async client instance."""
    if _client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama async client not initialized",
        )
    return _client


def get_request_context(request: Request) -> RequestContext:
    """Extract request context from FastAPI request."""
    return RequestContext(
        request_id=str(uuid.uuid4()),
        client_ip=get_remote_address(request),
        user_agent=request.headers.get("user-agent"),
        project_name=request.headers.get("x-project-name"),
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the health status of the API and underlying Ollama service.
    """
    is_healthy, error = check_service_health()
    ollama_status = "healthy" if is_healthy else f"unhealthy: {error}"

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        ollama_service=ollama_status,
        version="1.0.0",
    )


@app.get("/api/v1/models", response_model=ModelsResponse, tags=["Models"])
@limiter.limit("30/minute")
async def list_models(request: Request) -> ModelsResponse:
    """
    List available models.

    Returns a list of all models available in the Ollama service.
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

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("Error listing models")

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

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {exc!s}",
        ) from exc


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["Generation"])
@limiter.limit("60/minute")
async def generate(request: Request, generate_req: GenerateRequest) -> GenerateResponse:
    """
    Generate text from a prompt.

    Uses the specified model (or default) to generate text from the given prompt.
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()

    try:
        client = get_client()

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

        # Generate (async)
        result = await client.generate(
            prompt=generate_req.prompt,
            model=generate_req.model,
            system=generate_req.system,
            options=options,
            stream=generate_req.stream,
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

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("Error generating text")

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

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc!s}",
        ) from exc


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("60/minute")
async def chat(request: Request, chat_req: ChatRequest) -> ChatResponse:
    """
    Chat completion endpoint.

    Processes a conversation with multiple messages and returns the assistant's response.
    """
    ctx = get_request_context(request)
    start_time = time.perf_counter()

    try:
        client = get_client()

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

        # Chat (async)
        result = await client.chat(
            messages=messages, model=chat_req.model, options=options, stream=chat_req.stream
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

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.exception("Error in chat completion")

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

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {exc!s}",
        ) from exc


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            error_type=type(exc).__name__,
            request_id=get_request_context(request).request_id,
        ).model_dump(),
    )


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "service": "Shared Ollama Service API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }
