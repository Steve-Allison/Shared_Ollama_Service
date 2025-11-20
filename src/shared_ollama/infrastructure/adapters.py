"""Infrastructure adapters implementing application layer interfaces.

These adapters wrap infrastructure implementations (HTTP clients, loggers,
metrics collectors, image processors, caches, analytics, performance) to
satisfy the protocols defined in the application layer.

This enables dependency inversion: the application layer depends on interfaces,
not concrete implementations.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from shared_ollama.application.interfaces import ImageFormat
from shared_ollama.client.sync import GenerateOptions
from shared_ollama.telemetry.analytics import AnalyticsCollector
from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.performance import PerformanceCollector
from shared_ollama.telemetry.structured_logging import log_request_event

if TYPE_CHECKING:
    from shared_ollama.client import AsyncSharedOllamaClient
    from shared_ollama.infrastructure.image_cache import ImageCache
    from shared_ollama.infrastructure.image_processing import ImageProcessor


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

    async def generate(  # noqa: PLR0917
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        format: str | dict[str, Any] | None = None,  # noqa: A002
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Generate text from a prompt with tool calling support.

        Args:
            prompt: Text prompt for generation.
            model: Model name. Optional.
            system: System message. Optional.
            options: Generation options. Optional.
            stream: Whether to stream the response.
            format: Output format. Optional.
            tools: List of tools/functions the model can call (POML compatible). Optional.

        Returns:
            - dict with generation result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        # Convert options dict to GenerateOptions if needed
        generate_options: GenerateOptions | None = None
        if options:
            generate_options = GenerateOptions(
                temperature=options.get("temperature"),  # type: ignore[arg-type]
                top_p=options.get("top_p"),  # type: ignore[arg-type]
                top_k=options.get("top_k"),  # type: ignore[arg-type]
                repeat_penalty=options.get("repeat_penalty"),  # type: ignore[arg-type]
                max_tokens=options.get("num_predict"),  # type: ignore[arg-type]
                seed=options.get("seed"),  # type: ignore[arg-type]
                stop=options.get("stop"),  # type: ignore[arg-type]
            )

        if stream:
            stream_result = self._client.generate_stream(  # type: ignore[attr-defined]
                prompt=prompt,
                model=model,
                system=system,
                options=generate_options,
                format=format,  # type: ignore[arg-type]
                tools=tools,
            )
            return await self._resolve_stream_result(stream_result)
        # Type ignore needed because pyright doesn't see full method signature
        result = await self._client.generate(  # type: ignore[call-arg, misc]
            prompt=prompt,
            model=model,
            system=system,
            options=generate_options,
            stream=False,
            format=format,  # type: ignore[arg-type]
            tools=tools,
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

    async def chat(  # noqa: PLR0917
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
        images: list[str] | None = None,
        format: str | dict[str, Any] | None = None,  # noqa: A002
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Chat completion with multimodal and tool calling support.

        Uses Ollama's native API format:
        - messages: List of message dicts with 'role' and 'content' (string) keys
        - images: Optional list of base64-encoded image strings (Ollama's native format)
        - format: Optional output format ("json" or JSON schema dict)
        - tools: Optional list of tools/functions the model can call (POML compatible)

        Args:
            messages: List of message dicts with 'role' and optional 'content',
                'tool_calls', and 'tool_call_id' keys. Supports tool calling.
            model: Model name. Optional.
            options: Generation options. Optional.
            stream: Whether to stream the response.
            images: Optional list of base64-encoded image strings for vision models.
                This is Ollama's native format - images are passed as a separate parameter.
            format: Output format ("json" or JSON schema dict). Optional.
            tools: List of tools/functions the model can call (POML compatible). Optional.

        Returns:
            - dict with chat result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        # Convert options dict to GenerateOptions if needed
        generate_options: GenerateOptions | None = None
        if options:
            generate_options = GenerateOptions(
                temperature=options.get("temperature"),  # type: ignore[arg-type]
                top_p=options.get("top_p"),  # type: ignore[arg-type]
                top_k=options.get("top_k"),  # type: ignore[arg-type]
                repeat_penalty=options.get("repeat_penalty"),  # type: ignore[arg-type]
                max_tokens=options.get("num_predict"),  # type: ignore[arg-type]
                seed=options.get("seed"),  # type: ignore[arg-type]
                stop=options.get("stop"),  # type: ignore[arg-type]
            )

        if stream:
            stream_result = self._client.chat_stream(  # type: ignore[attr-defined]
                messages=messages,
                model=model,
                options=generate_options,
                images=images,
                format=format,
                tools=tools,
            )
            return await self._resolve_stream_result(stream_result)
        # Type ignore needed because pyright doesn't see full method signature
        return await self._client.chat(  # type: ignore[call-arg, misc]
            messages=messages,
            model=model,
            options=generate_options,  # type: ignore[arg-type]
            stream=False,
            images=images,  # type: ignore[arg-type]
            format=format,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )

    @staticmethod
    async def _resolve_stream_result(stream_result: Any) -> Any:
        """Return stream result, awaiting if upstream returned awaitable."""
        if inspect.isawaitable(stream_result):
            return await stream_result
        return stream_result

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

    @staticmethod
    def log_request(data: dict[str, Any]) -> None:
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

    @staticmethod
    def record_request(
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



class ImageProcessorAdapter:
    """Adapter that wraps ImageProcessor to implement ImageProcessorInterface.

    This adapter bridges the gap between the infrastructure image processing
    implementation and the application layer interface.
    """

    def __init__(self, processor: ImageProcessor) -> None:
        """Initialize the adapter.

        Args:
            processor: ImageProcessor instance to wrap.
        """
        self._processor = processor

    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Validate and parse image data URL.

        Args:
            data_url: Base64-encoded data URL.

        Returns:
            Tuple of (format, image_bytes).

        Raises:
            ValueError: If data URL is invalid.
        """
        return self._processor.validate_data_url(data_url)

    def process_image(
        self,
        data_url: str,
        target_format: ImageFormat = "jpeg",
    ) -> tuple[str, Any]:  # Returns (base64_string, ImageMetadata)
        """Process and optimize image for VLM model.

        Args:
            data_url: Base64-encoded data URL.
            target_format: Target image format ("jpeg", "png", or "webp").

        Returns:
            Tuple of (base64_string, metadata).

        Raises:
            ValueError: If image is invalid.
        """
        return self._processor.process_image(data_url, target_format=target_format)




class ImageCacheAdapter:
    """Adapter that wraps ImageCache to implement ImageCacheInterface.

    This adapter bridges the gap between the infrastructure image cache
    implementation and the application layer interface.
    """

    def __init__(self, cache: ImageCache) -> None:
        """Initialize the adapter.

        Args:
            cache: ImageCache instance to wrap.
        """
        self._cache = cache

    def get(
        self,
        data_url: str,
        target_format: ImageFormat,
    ) -> tuple[str, Any] | None:  # Returns (base64_string, ImageMetadata) | None
        """Get cached processed image.

        Args:
            data_url: Original data URL.
            target_format: Target format.

        Returns:
            Tuple of (base64_string, metadata) if cached and valid, None otherwise.
        """
        return self._cache.get(data_url, target_format)

    def put(
        self,
        data_url: str,
        target_format: ImageFormat,
        base64_string: str,
        metadata: Any,  # ImageMetadata
    ) -> None:
        """Cache processed image.

        Args:
            data_url: Original data URL.
            target_format: Target format.
            base64_string: Processed base64 string.
            metadata: Image metadata.
        """
        self._cache.put(data_url, target_format, base64_string, metadata)

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hits, misses, hit_rate, etc.).
        """
        return self._cache.get_stats()




class AnalyticsCollectorAdapter:
    """Adapter that wraps AnalyticsCollector to implement AnalyticsCollectorInterface.

    This adapter bridges the gap between the infrastructure analytics
    implementation and the application layer interface.

    Attributes:
        None. Uses the global AnalyticsCollector class.
    """

    @staticmethod
    def record_request_with_project(  # noqa: PLR0917
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        project: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a request metric with project tracking.

        Args:
            model: Model name or "system" for non-model operations.
            operation: Operation name (e.g., "generate", "chat", "vlm").
            latency_ms: Request latency in milliseconds.
            success: Whether the request succeeded.
            project: Project name from X-Project-Name header. Optional.
            error: Error name if request failed, None if succeeded.
        """
        AnalyticsCollector.record_request_with_project(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            project=project,
            error=error,
        )




class PerformanceCollectorAdapter:
    """Adapter that wraps PerformanceCollector to implement PerformanceCollectorInterface.

    This adapter bridges the gap between the infrastructure performance
    implementation and the application layer interface.

    Attributes:
        None. Uses the global PerformanceCollector class.
    """

    @staticmethod
    def record_performance(  # noqa: PLR0917
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: Any = None,  # GenerateResponse | dict[str, Any] | None
    ) -> None:
        """Record detailed performance metrics.

        Args:
            model: Model name used for the request.
            operation: Operation type (e.g., "generate", "chat", "vlm").
            total_latency_ms: Total request latency in milliseconds.
            success: Whether the request succeeded.
            response: GenerateResponse object or dictionary with timing data. Optional.
        """
        PerformanceCollector.record_performance(
            model=model,
            operation=operation,
            total_latency_ms=total_latency_ms,
            success=success,
            response=response,
        )
