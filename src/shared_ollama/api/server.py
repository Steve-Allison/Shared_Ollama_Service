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

import logging

from fastapi import FastAPI, Request

# Import modular components
from shared_ollama.api.lifespan import lifespan_context
from shared_ollama.api.middleware import setup_exception_handlers, setup_middleware
from shared_ollama.api.models import RequestContext
from shared_ollama.api.routes import (
    batch_router,
    chat_router,
    generation_router,
    system_router,
    vlm_router,
)

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
app.include_router(generation_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(vlm_router, prefix="/api/v1")
app.include_router(batch_router, prefix="/api/v1")


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


# All endpoints (generate, chat, vlm, batch) are now in modular route files:
# - routes/generation.py: /api/v1/generate
# - routes/chat.py: /api/v1/chat
# - routes/vlm.py: /api/v1/vlm, /api/v1/vlm/openai
# - routes/batch.py: /api/v1/batch/chat, /api/v1/batch/vlm
# - routes/system.py: /api/v1/health, /api/v1/models, /api/v1/queue/stats, /api/v1/metrics, /api/v1/performance/stats, /api/v1/analytics


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
