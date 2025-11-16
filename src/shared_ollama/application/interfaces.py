"""Interfaces (Protocols) for application layer dependencies.

These protocols define the contracts that infrastructure implementations
must satisfy. The application layer depends on these interfaces, not
concrete implementations, enabling dependency inversion.

All interfaces use Python 3.13+ Protocol for structural typing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationRequest,
    ModelInfo,
)


class OllamaClientInterface(Protocol):
    """Protocol for Ollama client implementations.

    Defines the interface that all Ollama client implementations must satisfy.
    This allows the application layer to work with any client implementation
    (sync, async, mock, etc.) without depending on concrete classes.
    """

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models.

        Returns:
            List of model dictionaries with name, size, modified_at keys.

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        ...

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        format: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Generate text from a prompt with tool calling support.

        Args:
            prompt: Text prompt for generation.
            model: Model name. Optional.
            system: System message. Optional.
            options: Generation options. Optional.
            stream: Whether to stream the response.
            format: Output format ("json" or JSON schema dict). Optional.
            tools: List of tools/functions the model can call (POML compatible). Optional.

        Returns:
            - dict with generation result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        images: list[str] | None = None,
        format: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Chat completion with multimodal and tool calling support.

        Args:
            messages: List of message dicts with 'role', optional 'content',
                'tool_calls', and 'tool_call_id' keys. Supports tool calling.
            model: Model name. Optional.
            options: Generation options. Optional.
            stream: Whether to stream the response.
            images: List of base64-encoded images (native Ollama format). Optional.
            format: Output format ("json" or JSON schema dict). Optional.
            tools: List of tools/functions the model can call (POML compatible). Optional.

        Returns:
            - dict with chat result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        ...

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy.

        Returns:
            True if service is healthy, False otherwise.
        """
        ...


class RequestLoggerInterface(Protocol):
    """Protocol for request logging implementations.

    Defines the interface for structured request logging. Allows different
    logging implementations (file, stdout, cloud, etc.) without coupling
    to specific implementations.
    """

    def log_request(self, data: dict[str, Any]) -> None:
        """Log a request event.

        Args:
            data: Dictionary with request event data. Must include:
                - event: Event type (e.g., "api_request")
                - status: "success" or "error"
                - request_id: Unique request identifier
                - operation: Operation name (e.g., "generate", "chat")
                - Additional fields as needed.
        """
        ...


class MetricsCollectorInterface(Protocol):
    """Protocol for metrics collection implementations.

    Defines the interface for collecting and recording metrics. Allows
    different metrics backends (Prometheus, StatsD, custom, etc.) without
    coupling to specific implementations.
    """

    def record_request(
        self,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record a request metric.

        Args:
            model: Model name or "system" for non-model operations.
            operation: Operation name (e.g., "generate", "chat", "list_models").
            latency_ms: Request latency in milliseconds.
            success: Whether the request succeeded.
            error: Error name if request failed, None if succeeded.
        """
        ...

