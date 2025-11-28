"""Dependency injection for FastAPI endpoints.

This module provides FastAPI dependencies for use cases and infrastructure
components, enabling dependency injection and removing global state. All
dependencies are injected via FastAPI's Depends() system, following Clean
Architecture principles.

Design Principles:
    - Dependency Injection: All components injected via FastAPI Depends()
    - No Global State: Dependencies stored in module-level variables but
      accessed only through dependency functions
    - Lifecycle Management: Dependencies initialized during lifespan startup
    - Error Handling: Returns 503 if dependencies not initialized

Key Dependencies:
    - Use Cases: GenerateUseCase, ChatUseCase, VLMUseCase, ListModelsUseCase
    - Batch Use Cases: BatchChatUseCase, BatchVLMUseCase
    - Infrastructure: Client, logger, metrics, queues, image processors
    - Request Context: Extracts request_id, client_ip, project_name from headers

Dependency Flow:
    1. Lifespan startup initializes adapters and queues
    2. set_dependencies() stores global instances
    3. get_*() functions retrieve instances (raise 503 if not initialized)
    4. get_*_use_case() functions construct use cases with injected dependencies
    5. FastAPI Depends() wires everything together
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, TypeVar

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel

from shared_ollama.api.limits import MAX_REQUEST_BODY_BYTES
from shared_ollama.api.models import RequestContext
from shared_ollama.application.batch_use_cases import BatchChatUseCase, BatchVLMUseCase
from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import get_allowed_models, is_model_allowed
from shared_ollama.infrastructure.adapters import (
    AnalyticsCollectorAdapter,
    AsyncOllamaClientAdapter,
    ImageCacheAdapter,
    ImageProcessorAdapter,
    MetricsCollectorAdapter,
    PerformanceCollectorAdapter,
    RequestLoggerAdapter,
)
from shared_ollama.infrastructure.config import settings

logger = logging.getLogger(__name__)

# Generic type for request models
T = TypeVar("T", bound=BaseModel)

# Global instances (initialized in lifespan)
_client_adapter: AsyncOllamaClientAdapter | None = None
_logger_adapter: RequestLoggerAdapter | None = None
_metrics_adapter: MetricsCollectorAdapter | None = None
_analytics_adapter: AnalyticsCollectorAdapter | None = None
_performance_adapter: PerformanceCollectorAdapter | None = None
_chat_queue: RequestQueue | None = None
_vlm_queue: RequestQueue | None = None
_image_processor_adapter: ImageProcessorAdapter | None = None
_image_cache_adapter: ImageCacheAdapter | None = None


def validate_dependencies() -> dict[str, bool]:
    """Validate that all required dependencies are initialized.

    Checks all global dependency instances to verify they have been
    properly initialized during lifespan startup. Returns a dictionary
    mapping dependency names to their initialization status.

    Returns:
        Dictionary with dependency names as keys and boolean initialization
        status as values. True means initialized, False means not initialized.

    Example:
        ```python
        status = validate_dependencies()
        if not all(status.values()):
            missing = [k for k, v in status.items() if not v]
            raise RuntimeError(f"Missing dependencies: {missing}")
        ```

    Note:
        This function does not raise exceptions. Use the returned dictionary
        to determine which dependencies are missing and handle accordingly.
    """
    return {
        "client_adapter": _client_adapter is not None,
        "logger_adapter": _logger_adapter is not None,
        "metrics_adapter": _metrics_adapter is not None,
        "chat_queue": _chat_queue is not None,
        "vlm_queue": _vlm_queue is not None,
        "image_processor_adapter": _image_processor_adapter is not None,
        "image_cache_adapter": _image_cache_adapter is not None,
        # Optional adapters (may be None by design)
        "analytics_adapter": True,  # Optional, always "valid"
        "performance_adapter": True,  # Optional, always "valid"
    }


def set_dependencies(
    client_adapter: AsyncOllamaClientAdapter,
    logger_adapter: RequestLoggerAdapter,
    metrics_adapter: MetricsCollectorAdapter,
    chat_queue: RequestQueue,
    vlm_queue: RequestQueue,
    image_processor_adapter: ImageProcessorAdapter,
    image_cache_adapter: ImageCacheAdapter,
    analytics_adapter: AnalyticsCollectorAdapter | None = None,
    performance_adapter: PerformanceCollectorAdapter | None = None,
) -> None:
    """Set global dependencies (called during lifespan startup).

    Stores infrastructure adapters and queues in module-level variables for
    access by FastAPI dependency functions. This function is called once
    during application startup in lifespan_context().

    Args:
        client_adapter: Ollama client adapter for making requests to Ollama service.
        logger_adapter: Request logger adapter for structured logging.
        metrics_adapter: Metrics collector adapter for basic metrics.
        chat_queue: Chat request queue for concurrency control (text-only requests).
        vlm_queue: VLM request queue for concurrency control (vision-language requests).
        image_processor_adapter: Image processor adapter for VLM image processing.
        image_cache_adapter: Image cache adapter for caching processed images.
        analytics_adapter: Analytics collector adapter for project-based tracking.
            None if analytics collection is disabled.
        performance_adapter: Performance collector adapter for detailed metrics.
            None if detailed performance tracking is disabled.

    Note:
        This function should only be called once during application startup.
        All dependencies must be initialized before calling this function.
        Dependencies are stored in module-level variables and accessed via
        get_*() functions which raise 503 if not initialized.
    """
    global _client_adapter, _logger_adapter, _metrics_adapter, _analytics_adapter
    global _performance_adapter, _chat_queue, _vlm_queue
    global _image_processor_adapter, _image_cache_adapter
    _client_adapter = client_adapter
    _logger_adapter = logger_adapter
    _metrics_adapter = metrics_adapter
    _analytics_adapter = analytics_adapter
    _performance_adapter = performance_adapter
    _chat_queue = chat_queue
    _vlm_queue = vlm_queue
    _image_processor_adapter = image_processor_adapter
    _image_cache_adapter = image_cache_adapter


def get_client_adapter() -> AsyncOllamaClientAdapter:
    """Get the Ollama client adapter.

    Returns:
        AsyncOllamaClientAdapter instance.

    Raises:
        HTTPException: If adapter not initialized.
    """
    if _client_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama client adapter not initialized",
        )
    return _client_adapter


def get_logger_adapter() -> RequestLoggerAdapter:
    """Get the request logger adapter.

    Returns:
        RequestLoggerAdapter instance.

    Raises:
        HTTPException: If adapter not initialized.
    """
    if _logger_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Request logger adapter not initialized",
        )
    return _logger_adapter


def get_metrics_adapter() -> MetricsCollectorAdapter:
    """Get the metrics collector adapter.

    Returns:
        MetricsCollectorAdapter instance.

    Raises:
        HTTPException: If adapter not initialized.
    """
    if _metrics_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collector adapter not initialized",
        )
    return _metrics_adapter


def get_chat_queue() -> RequestQueue:
    """Get the chat request queue.

    Returns:
        RequestQueue instance for chat requests.

    Raises:
        HTTPException: If queue not initialized.
    """
    if _chat_queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat request queue not initialized",
        )
    return _chat_queue


def get_vlm_queue() -> RequestQueue:
    """Get the VLM request queue.

    Returns:
        RequestQueue instance for VLM requests.

    Raises:
        HTTPException: If queue not initialized.
    """
    if _vlm_queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VLM request queue not initialized",
        )
    return _vlm_queue


def get_image_processor() -> ImageProcessorAdapter:
    """Get the image processor adapter.

    Returns:
        ImageProcessorAdapter instance.

    Raises:
        HTTPException: If processor not initialized.
    """
    if _image_processor_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image processor not initialized",
        )
    return _image_processor_adapter


def get_image_cache() -> ImageCacheAdapter:
    """Get the image cache adapter.

    Returns:
        ImageCacheAdapter instance.

    Raises:
        HTTPException: If cache not initialized.
    """
    if _image_cache_adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image cache not initialized",
        )
    return _image_cache_adapter


def get_request_context(request: Request) -> RequestContext:  # type: ignore[valid-type]
    """Extract (or reuse) request context from FastAPI request.

    Extracts or creates request context from FastAPI request, ensuring the same
    context instance is reused throughout the lifetime of a single request. This
    ensures all logging and metrics share the identical request_id, client_ip,
    and project_name.

    The context is stored in request.state to avoid recomputation and ensure
    consistency across middleware, dependencies, and route handlers.

    Args:
        request: FastAPI Request object containing headers and state.

    Returns:
        RequestContext object containing:
            - request_id: Unique request identifier (UUID v4)
            - client_ip: Client IP address (from X-Forwarded-For or remote address)
            - user_agent: User agent string from headers
            - project_name: Project name from X-Project-Name header (optional)

    Note:
        Request context is cached in request.state after first access. This
        ensures consistent request_id across all logging and metrics for a
        single HTTP request.
    """
    from slowapi.util import get_remote_address

    ctx: RequestContext | None = getattr(request.state, "request_context", None)
    if ctx is None:
        ctx = RequestContext(
            request_id=str(uuid.uuid4()),
            client_ip=get_remote_address(request),
            user_agent=request.headers.get("user-agent"),
            project_name=request.headers.get("x-project-name"),
        )
        request.state.request_context = ctx
    return ctx


# FastAPI dependency annotations
def get_generate_use_case(
    client_adapter: Annotated[AsyncOllamaClientAdapter, Depends(get_client_adapter)],
    logger_adapter: Annotated[RequestLoggerAdapter, Depends(get_logger_adapter)],
    metrics_adapter: Annotated[MetricsCollectorAdapter, Depends(get_metrics_adapter)],
) -> GenerateUseCase:
    """Get GenerateUseCase instance.

    Args:
        client_adapter: Ollama client adapter (injected).
        logger_adapter: Request logger adapter (injected).
        metrics_adapter: Metrics collector adapter (injected).

    Returns:
        GenerateUseCase instance.
    """
    return GenerateUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )


def get_chat_use_case(
    client_adapter: Annotated[AsyncOllamaClientAdapter, Depends(get_client_adapter)],
    logger_adapter: Annotated[RequestLoggerAdapter, Depends(get_logger_adapter)],
    metrics_adapter: Annotated[MetricsCollectorAdapter, Depends(get_metrics_adapter)],
) -> ChatUseCase:
    """Get ChatUseCase instance.

    Args:
        client_adapter: Ollama client adapter (injected).
        logger_adapter: Request logger adapter (injected).
        metrics_adapter: Metrics collector adapter (injected).

    Returns:
        ChatUseCase instance.
    """
    return ChatUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )


def get_list_models_use_case(
    client_adapter: Annotated[AsyncOllamaClientAdapter, Depends(get_client_adapter)],
    logger_adapter: Annotated[RequestLoggerAdapter, Depends(get_logger_adapter)],
    metrics_adapter: Annotated[MetricsCollectorAdapter, Depends(get_metrics_adapter)],
) -> ListModelsUseCase:
    """Get ListModelsUseCase instance.

    Args:
        client_adapter: Ollama client adapter (injected).
        logger_adapter: Request logger adapter (injected).
        metrics_adapter: Metrics collector adapter (injected).

    Returns:
        ListModelsUseCase instance.
    """
    return ListModelsUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
    )


def get_vlm_use_case(
    client_adapter: Annotated[AsyncOllamaClientAdapter, Depends(get_client_adapter)],
    logger_adapter: Annotated[RequestLoggerAdapter, Depends(get_logger_adapter)],
    metrics_adapter: Annotated[MetricsCollectorAdapter, Depends(get_metrics_adapter)],
    image_processor_adapter: Annotated[ImageProcessorAdapter, Depends(get_image_processor)],
    image_cache_adapter: Annotated[ImageCacheAdapter, Depends(get_image_cache)],
) -> VLMUseCase:
    """Get VLMUseCase instance.

    Args:
        client_adapter: Ollama client adapter (injected).
        logger_adapter: Request logger adapter (injected).
        metrics_adapter: Metrics collector adapter (injected).
        image_processor_adapter: Image processor adapter (injected).
        image_cache_adapter: Image cache adapter (injected).

    Returns:
        VLMUseCase instance.
    """
    # Get optional analytics and performance adapters
    analytics_adapter = _analytics_adapter
    performance_adapter = _performance_adapter

    return VLMUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
        image_processor=image_processor_adapter,
        image_cache=image_cache_adapter,
        analytics=analytics_adapter,
        performance=performance_adapter,
    )


def get_batch_chat_use_case(
    chat_use_case: Annotated[ChatUseCase, Depends(get_chat_use_case)],
) -> BatchChatUseCase:
    """Get BatchChatUseCase instance.

    Args:
        chat_use_case: Chat use case (injected).

    Returns:
        BatchChatUseCase instance.
    """
    return BatchChatUseCase(
        chat_use_case=chat_use_case,
        max_concurrent=settings.batch.chat_max_concurrent,
    )


def get_batch_vlm_use_case(
    vlm_use_case: Annotated[VLMUseCase, Depends(get_vlm_use_case)],
) -> BatchVLMUseCase:
    """Get BatchVLMUseCase instance.

    Args:
        vlm_use_case: VLM use case (injected).

    Returns:
        BatchVLMUseCase instance.
    """
    return BatchVLMUseCase(
        vlm_use_case=vlm_use_case,
        max_concurrent=settings.batch.vlm_max_concurrent,
    )


# ============================================================================
# Request Parsing & Validation Helpers
# ============================================================================


async def parse_request_json(request: Request, model_cls: type[T]) -> T:
    """Parse and validate request JSON into a Pydantic model.

    Generic helper that handles JSON parsing and Pydantic validation with
    consistent error handling across all route handlers.

    Args:
        request: FastAPI Request object containing JSON body.
        model_cls: Pydantic model class to validate against.

    Returns:
        Validated Pydantic model instance.

    Raises:
        HTTPException: 422 Unprocessable Entity if JSON is invalid or
            validation fails. Error detail includes specific validation message.

    Example:
        ```python
        api_req = await parse_request_json(request, ChatRequest)
        ```
    """
    try:
        body_bytes = await request.body()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unable to read request body.",
        ) from exc

    if not body_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Request body is required.",
        )

    if len(body_bytes) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "request_too_large",
                "message": (
                    f"Request body is {len(body_bytes):,} bytes but the limit is "
                    f"{MAX_REQUEST_BODY_BYTES:,} bytes. Split or compress the payload and retry."
                ),
                "limit_bytes": MAX_REQUEST_BODY_BYTES,
                "actual_bytes": len(body_bytes),
            },
        )

    try:
        body = json.loads(body_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid JSON in request body: {exc!s}",
        ) from exc

    try:
        return model_cls(**body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Request validation failed: {exc!s}",
        ) from exc


def validate_model_allowed(model_name: str | None) -> None:
    """Validate that the requested model is allowed for the current hardware profile.

    Checks the model name against the allowed models list configured for
    the current hardware tier. Raises HTTPException if the model is not allowed.

    Args:
        model_name: Model name to validate. If None, validation passes (uses default).

    Raises:
        HTTPException: 400 Bad Request if model is not in the allowed list.
            Error detail includes the list of allowed models.

    Example:
        ```python
        validate_model_allowed(api_req.model)
        # Raises HTTPException if model not allowed
        ```
    """
    if model_name and not is_model_allowed(model_name):
        allowed = get_allowed_models()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{model_name}' is not supported on this hardware profile. "
                f"Allowed models: {', '.join(sorted(allowed))}"
            ),
        )
