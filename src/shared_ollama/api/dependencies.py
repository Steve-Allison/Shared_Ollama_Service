"""Dependency injection for FastAPI endpoints.

This module provides FastAPI dependencies for use cases and infrastructure
components, enabling dependency injection and removing global state.

All dependencies are injected via FastAPI's Depends() system, following
Clean Architecture principles.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, status

from shared_ollama.api.models import RequestContext
from shared_ollama.application.batch_use_cases import BatchChatUseCase, BatchVLMUseCase
from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)
from shared_ollama.application.vlm_use_cases import VLMUseCase
from shared_ollama.infrastructure.config import settings

if TYPE_CHECKING:
    from fastapi import Request

    from shared_ollama.core.queue import RequestQueue
    from shared_ollama.infrastructure.adapters import (
        AnalyticsCollectorAdapter,
        AsyncOllamaClientAdapter,
        ImageCacheAdapter,
        ImageProcessorAdapter,
        MetricsCollectorAdapter,
        PerformanceCollectorAdapter,
        RequestLoggerAdapter,
    )
else:
    from shared_ollama.core.queue import RequestQueue
    from shared_ollama.infrastructure.adapters import (
        AnalyticsCollectorAdapter,
        AsyncOllamaClientAdapter,
        ImageCacheAdapter,
        ImageProcessorAdapter,
        MetricsCollectorAdapter,
        PerformanceCollectorAdapter,
        RequestLoggerAdapter,
    )

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

    Args:
        client_adapter: Ollama client adapter.
        logger_adapter: Request logger adapter.
        metrics_adapter: Metrics collector adapter.
        chat_queue: Chat request queue instance.
        vlm_queue: VLM request queue instance.
        image_processor_adapter: Image processor adapter for VLM.
        image_cache_adapter: Image cache adapter for VLM.
        analytics_adapter: Analytics collector adapter. Optional.
        performance_adapter: Performance collector adapter. Optional.
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

    Ensures the same context instance is reused throughout the lifetime of a single
    request so all logging shares the identical request_id.

    Args:
        request: FastAPI Request object.

    Returns:
        RequestContext with request_id, client_ip, user_agent, and project_name.
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
