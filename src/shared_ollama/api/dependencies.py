"""Dependency injection for FastAPI endpoints.

This module provides FastAPI dependencies for use cases and infrastructure
components, enabling dependency injection and removing global state.

All dependencies are injected via FastAPI's Depends() system, following
Clean Architecture principles.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from shared_ollama.api.models import RequestContext
from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)
from shared_ollama.core.queue import RequestQueue
from shared_ollama.infrastructure.adapters import (
    AsyncOllamaClientAdapter,
    MetricsCollectorAdapter,
    RequestLoggerAdapter,
)

# Global instances (initialized in lifespan)
_client_adapter: AsyncOllamaClientAdapter | None = None
_logger_adapter: RequestLoggerAdapter | None = None
_metrics_adapter: MetricsCollectorAdapter | None = None
_queue: RequestQueue | None = None


def set_dependencies(
    client_adapter: AsyncOllamaClientAdapter,
    logger_adapter: RequestLoggerAdapter,
    metrics_adapter: MetricsCollectorAdapter,
    queue: RequestQueue,
) -> None:
    """Set global dependencies (called during lifespan startup).

    Args:
        client_adapter: Ollama client adapter.
        logger_adapter: Request logger adapter.
        metrics_adapter: Metrics collector adapter.
        queue: Request queue instance.
    """
    global _client_adapter, _logger_adapter, _metrics_adapter, _queue
    _client_adapter = client_adapter
    _logger_adapter = logger_adapter
    _metrics_adapter = metrics_adapter
    _queue = queue


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


def get_queue() -> RequestQueue:
    """Get the request queue.

    Returns:
        RequestQueue instance.

    Raises:
        HTTPException: If queue not initialized.
    """
    if _queue is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Request queue not initialized",
        )
    return _queue


def get_request_context(request: Request) -> RequestContext:
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
