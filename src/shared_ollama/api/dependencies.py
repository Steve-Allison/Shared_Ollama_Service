"""Dependency injection for FastAPI endpoints.

This module provides FastAPI dependencies for use cases and infrastructure
components, enabling dependency injection and removing global state.

All dependencies are injected via FastAPI's Depends() system, following
Clean Architecture principles.
"""

from __future__ import annotations

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
from shared_ollama.core.config import settings

if TYPE_CHECKING:
    from fastapi import Request

    from shared_ollama.core.queue import RequestQueue
    from shared_ollama.infrastructure.adapters import (
        AsyncOllamaClientAdapter,
        MetricsCollectorAdapter,
        RequestLoggerAdapter,
    )
    from shared_ollama.infrastructure.image_cache import ImageCache
    from shared_ollama.infrastructure.image_processing import ImageProcessor
else:
    from shared_ollama.core.queue import RequestQueue
    from shared_ollama.infrastructure.adapters import (
        AsyncOllamaClientAdapter,
        MetricsCollectorAdapter,
        RequestLoggerAdapter,
    )
    from shared_ollama.infrastructure.image_cache import ImageCache
    from shared_ollama.infrastructure.image_processing import ImageProcessor

# Global instances (initialized in lifespan)
_client_adapter: AsyncOllamaClientAdapter | None = None
_logger_adapter: RequestLoggerAdapter | None = None
_metrics_adapter: MetricsCollectorAdapter | None = None
_chat_queue: RequestQueue | None = None
_vlm_queue: RequestQueue | None = None
_image_processor: ImageProcessor | None = None
_image_cache: ImageCache | None = None


def set_dependencies(
    client_adapter: AsyncOllamaClientAdapter,
    logger_adapter: RequestLoggerAdapter,
    metrics_adapter: MetricsCollectorAdapter,
    chat_queue: RequestQueue,
    vlm_queue: RequestQueue,
    image_processor: ImageProcessor,
    image_cache: ImageCache,
) -> None:
    """Set global dependencies (called during lifespan startup).

    Args:
        client_adapter: Ollama client adapter.
        logger_adapter: Request logger adapter.
        metrics_adapter: Metrics collector adapter.
        chat_queue: Chat request queue instance.
        vlm_queue: VLM request queue instance.
        image_processor: Image processor for VLM.
        image_cache: Image cache for VLM.
    """
    global _client_adapter, _logger_adapter, _metrics_adapter, _chat_queue, _vlm_queue, _image_processor, _image_cache
    _client_adapter = client_adapter
    _logger_adapter = logger_adapter
    _metrics_adapter = metrics_adapter
    _chat_queue = chat_queue
    _vlm_queue = vlm_queue
    _image_processor = image_processor
    _image_cache = image_cache


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


def get_image_processor() -> ImageProcessor:
    """Get the image processor.

    Returns:
        ImageProcessor instance.

    Raises:
        HTTPException: If processor not initialized.
    """
    if _image_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image processor not initialized",
        )
    return _image_processor


def get_image_cache() -> ImageCache:
    """Get the image cache.

    Returns:
        ImageCache instance.

    Raises:
        HTTPException: If cache not initialized.
    """
    if _image_cache is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image cache not initialized",
        )
    return _image_cache


def get_request_context(request: Request) -> RequestContext:  # type: ignore[valid-type]
    """Extract request context from FastAPI request.

    Args:
        request: FastAPI Request object.

    Returns:
        RequestContext with request_id, client_ip, user_agent, and project_name.
    """
    import uuid

    from slowapi.util import get_remote_address

    return RequestContext(
        request_id=str(uuid.uuid4()),
        client_ip=get_remote_address(request),
        user_agent=request.headers.get("user-agent"),
        project_name=request.headers.get("x-project-name"),
    )


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
    image_processor: Annotated[ImageProcessor, Depends(get_image_processor)],
    image_cache: Annotated[ImageCache, Depends(get_image_cache)],
) -> VLMUseCase:
    """Get VLMUseCase instance.

    Args:
        client_adapter: Ollama client adapter (injected).
        logger_adapter: Request logger adapter (injected).
        metrics_adapter: Metrics collector adapter (injected).
        image_processor: Image processor (injected).
        image_cache: Image cache (injected).

    Returns:
        VLMUseCase instance.
    """
    return VLMUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
        image_processor=image_processor,
        image_cache=image_cache,
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
