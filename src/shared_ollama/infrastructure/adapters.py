"""Infrastructure adapters implementing application layer interfaces.

These adapters wrap infrastructure implementations (HTTP clients, loggers,
metrics collectors) to satisfy the protocols defined in the application layer.

This enables dependency inversion: the application layer depends on interfaces,
not concrete implementations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from shared_ollama.application.interfaces import (
    MetricsCollectorInterface,
    OllamaClientInterface,
    RequestLoggerInterface,
)
from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient
from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.structured_logging import log_request_event


class AsyncOllamaClientAdapter:
    """Adapter that wraps AsyncSharedOllamaClient to implement OllamaClientInterface.

    This adapter bridges the gap between the infrastructure HTTP client
    implementation and the application layer interface. It translates between
    domain concepts and HTTP client calls.

    Attributes:
        _client: The underlying AsyncSharedOllamaClient instance.
    """

    def __init__(self, client: AsyncSharedOllamaClient) -> None:
        """Initialize the adapter.

        Args:
            client: AsyncSharedOllamaClient instance to wrap.
        """
        self._client = client

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models.

        Returns:
            List of model dictionaries with name, size, modified_at keys.

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        return await self._client.list_models()

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        format: str | dict[str, Any] | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Generate text from a prompt.

        Args:
            prompt: Text prompt for generation.
            model: Model name. Optional.
            system: System message. Optional.
            options: Generation options. Optional.
            stream: Whether to stream the response.
            format: Output format. Optional.

        Returns:
            - dict with generation result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        # Convert options dict to GenerateOptions if needed
        from shared_ollama.client.sync import GenerateOptions

        generate_options: GenerateOptions | None = None
        if options:
            generate_options = GenerateOptions(
                temperature=options.get("temperature"),
                top_p=options.get("top_p"),
                top_k=options.get("top_k"),
                repeat_penalty=options.get("repeat_penalty"),
                max_tokens=options.get("num_predict"),
                seed=options.get("seed"),
                stop=options.get("stop"),
            )

        if stream:
            return self._client.generate_stream(
                prompt=prompt,
                model=model,
                system=system,
                options=generate_options,
            )
        else:
            result = await self._client.generate(
                prompt=prompt,
                model=model,
                system=system,
                options=generate_options,
                stream=False,
                format=format,
            )
            # Convert GenerateResponse to dict (preserving all fields)
            return {
                "text": result.text,
                "model": result.model,
                "prompt_eval_count": result.prompt_eval_count,
                "eval_count": result.eval_count,
                "total_duration": result.total_duration,
                "load_duration": result.load_duration,
            }

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model name. Optional.
            options: Generation options. Optional.
            stream: Whether to stream the response.

        Returns:
            - dict with chat result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        # Convert options dict to GenerateOptions if needed
        from shared_ollama.client.sync import GenerateOptions

        generate_options: GenerateOptions | None = None
        if options:
            generate_options = GenerateOptions(
                temperature=options.get("temperature"),
                top_p=options.get("top_p"),
                top_k=options.get("top_k"),
                repeat_penalty=options.get("repeat_penalty"),
                max_tokens=options.get("num_predict"),
                seed=options.get("seed"),
                stop=options.get("stop"),
            )

        if stream:
            return self._client.chat_stream(
                messages=messages,
                model=model,
                options=generate_options,
            )
        else:
            return await self._client.chat(
                messages=messages,
                model=model,
                options=generate_options,
                stream=False,
            )

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy.

        Returns:
            True if service is healthy, False otherwise.
        """
        return await self._client.health_check()


class RequestLoggerAdapter:
    """Adapter that wraps structured logging to implement RequestLoggerInterface.

    This adapter bridges the gap between the infrastructure logging
    implementation and the application layer interface.

    Attributes:
        None. Uses the global log_request_event function.
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
        log_request_event(data)


class MetricsCollectorAdapter:
    """Adapter that wraps MetricsCollector to implement MetricsCollectorInterface.

    This adapter bridges the gap between the infrastructure metrics
    implementation and the application layer interface.

    Attributes:
        None. Uses the global MetricsCollector class.
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
        MetricsCollector.record_request(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

