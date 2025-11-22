"""Infrastructure adapters implementing application layer interfaces.

This module provides adapter implementations that wrap concrete infrastructure
components (HTTP clients, loggers, metrics collectors, image processors, caches,
analytics, performance) to satisfy the protocols defined in the application layer.

Design Principles:
    - Adapter Pattern: Wraps concrete implementations to match interfaces
    - Dependency Inversion: Application depends on interfaces, not implementations
    - Translation: Converts between infrastructure and application formats
    - Delegation: Delegates to underlying infrastructure components

Key Adapters:
    - AsyncOllamaClientAdapter: Wraps AsyncSharedOllamaClient for OllamaClientInterface
    - ImageProcessorAdapter: Wraps ImageProcessor for ImageProcessorInterface
    - ImageCacheAdapter: Wraps ImageCache for ImageCacheInterface
    - RequestLoggerAdapter: Wraps structured logging for RequestLoggerInterface
    - MetricsCollectorAdapter: Wraps MetricsCollector for MetricsCollectorInterface
    - AnalyticsCollectorAdapter: Wraps AnalyticsCollector for AnalyticsCollectorInterface
    - PerformanceCollectorAdapter: Wraps PerformanceCollector for PerformanceCollectorInterface

Note:
    Adapters enable the application layer to work with any infrastructure
    implementation without coupling. They translate between domain/application
    formats and infrastructure-specific formats.
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
    application format (dicts) and infrastructure format (GenerateResponse objects,
    async iterators).

    The adapter handles:
        - Format conversion (GenerateResponse -> dict)
        - Options transformation (dict -> GenerateOptions)
        - Streaming result resolution
        - Error propagation

    Attributes:
        _client: The underlying AsyncSharedOllamaClient instance. All operations
            delegate to this client.

    Note:
        This adapter implements OllamaClientInterface, allowing the application
        layer to use any HTTP client implementation without coupling.
    """

    def __init__(self, client: AsyncSharedOllamaClient) -> None:
        """Initialize the adapter.

        Args:
            client: AsyncSharedOllamaClient instance to wrap. This client handles
                all HTTP communication with the Ollama service.
        """
        self._client = client

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models.

        Delegates to underlying client and returns models in dict format.

        Returns:
            List of model dictionaries. Each dict contains:
                - name: Model name identifier (str)
                - size: Model size in bytes (int, optional)
                - modified_at: ISO timestamp of last modification (str, optional)
                - Additional model metadata as available

        Raises:
            ConnectionError: If Ollama service is unavailable or unreachable.
            Exception: For other infrastructure or network errors.

        Note:
            This method directly delegates to the client without transformation,
            as the client already returns dict format.
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
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Generate text from a prompt with tool calling support.

        Converts application format (dict options) to infrastructure format
        (GenerateOptions object), delegates to client, and converts response
        back to application format (dict).

        Args:
            prompt: Text prompt for generation. Must not be empty.
            model: Model name identifier. None uses service default model.
            system: System message to set model behavior. None means no
                system message.
            options: Generation options dictionary. Keys may include:
                - temperature: float (0.0-2.0)
                - top_p: float (0.0-1.0)
                - top_k: int (>= 1)
                - num_predict: int (max tokens, >= 1)
                - seed: int (random seed)
                - stop: list[str] (stop sequences)
                None means use model defaults.
            stream: Whether to stream the response incrementally. True returns
                AsyncIterator, False returns dict.
            format: Output format specification. Can be "json", dict schema,
                or None. None means default text output.
            tools: List of tool definitions for function calling. POML compatible.
                None means no tool calling.

        Returns:
            If stream=False: dict with generation result containing:
                - text: Generated text (str)
                - model: Model name used (str)
                - prompt_eval_count: Prompt tokens evaluated (int)
                - eval_count: Generation tokens produced (int)
                - total_duration: Total generation time in nanoseconds (int)
                - load_duration: Model load time in nanoseconds (int)
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with metrics.

        Raises:
            ConnectionError: If Ollama service is unavailable or unreachable.
            ValueError: If prompt is empty or options are invalid.
            Exception: For other infrastructure or network errors.

        Note:
            Options dict is converted to GenerateOptions object for the client.
            Response is converted from GenerateResponse object to dict for the
            application layer.
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

        Converts application format (dict options) to infrastructure format
        (GenerateOptions object), delegates to client, and returns response
        in application format. Uses Ollama's native API format where images
        are passed as a separate parameter.

        Args:
            messages: List of message dictionaries forming conversation history.
                Each message dict contains:
                - role: str ("user", "assistant", "system", "tool")
                - content: str (optional, text content)
                - tool_calls: list[dict] (optional, for assistant messages)
                - tool_call_id: str (optional, for tool response messages)
            model: Model name identifier. None uses service default model.
            options: Generation options dictionary. Same format as generate().
                None means use model defaults.
            stream: Whether to stream the response incrementally. True returns
                AsyncIterator, False returns dict.
            images: List of base64-encoded image data URLs (Ollama's native format).
                Images are attached to the last user message. None means no images.
                This is Ollama's native format - images are passed separately,
                not embedded in message content.
            format: Output format specification. Can be "json", dict schema,
                or None. None means default text output.
            tools: List of tool definitions for function calling. POML compatible.
                None means no tool calling.

        Returns:
            If stream=False: dict with chat result (message, model, metrics).
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with complete metrics.

        Raises:
            ConnectionError: If Ollama service is unavailable or unreachable.
            ValueError: If messages are invalid or options are invalid.
            Exception: For other infrastructure or network errors.

        Note:
            This method uses Ollama's native format where images are passed as
            a separate parameter, not embedded in message content (unlike
            OpenAI format). Options dict is converted to GenerateOptions object.
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
        """Resolve stream result, awaiting if upstream returned awaitable.

        Handles cases where the client may return either an async iterator
        directly or an awaitable that resolves to an async iterator.

        Args:
            stream_result: Stream result from client. May be AsyncIterator
                directly or awaitable that resolves to AsyncIterator.

        Returns:
            AsyncIterator[dict[str, Any]] for streaming responses.

        Note:
            This method handles both synchronous and asynchronous stream
            results from the client implementation.
        """
        if inspect.isawaitable(stream_result):
            return await stream_result
        return stream_result

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy and reachable.

        Delegates to underlying client's health check method.

        Returns:
            True if service is healthy and reachable, False otherwise.

        Note:
            This method directly delegates to the client without transformation.
            The client handles the actual health check logic.
        """
        return await self._client.health_check()


class RequestLoggerAdapter:
    """Adapter that wraps structured logging to implement RequestLoggerInterface.

    This adapter bridges the gap between the infrastructure structured logging
    implementation and the application layer interface. Delegates to the global
    log_request_event function.

    Attributes:
        None. Uses the global log_request_event function from structured_logging.

    Note:
        This is a static adapter - no instance state is maintained. All
        logging is delegated to the global log_request_event function.
    """

    @staticmethod
    def log_request(data: dict[str, Any]) -> None:
        """Log a request event with structured data.

        Delegates to the global log_request_event function for structured
        request logging.

        Args:
            data: Dictionary with request event data. Required keys:
                - event: Event type identifier (e.g., "api_request")
                - status: Request status ("success" or "error")
                - request_id: Unique request identifier for tracing
                - operation: Operation name (e.g., "generate", "chat", "vlm")
            Optional keys may include model, client_ip, project_name, latency_ms,
            error_type, error_message, and operation-specific fields.

        Note:
            This method delegates to log_request_event which handles the actual
            logging implementation (file, stdout, cloud services, etc.).
        """
        log_request_event(data)


class MetricsCollectorAdapter:
    """Adapter that wraps MetricsCollector to implement MetricsCollectorInterface.

    This adapter bridges the gap between the infrastructure metrics collection
    implementation and the application layer interface. Delegates to the global
    MetricsCollector class.

    Attributes:
        None. Uses the global MetricsCollector class from telemetry.metrics.

    Note:
        This is a static adapter - no instance state is maintained. All
        metrics collection is delegated to the global MetricsCollector class.
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

        Delegates to the global MetricsCollector for metrics recording.

        Args:
            model: Model name identifier. Use "system" for non-model operations
                (e.g., "list_models", health checks).
            operation: Operation name identifier. Examples: "generate", "chat",
                "vlm", "list_models".
            latency_ms: Request latency in milliseconds. Measured from request
                start to completion.
            success: Whether the request succeeded. True for successful requests,
                False for errors.
            error: Error type name if request failed. None if succeeded.
                Examples: "ValueError", "ConnectionError", "TimeoutError".

        Note:
            This method delegates to MetricsCollector.record_request which handles
            the actual metrics collection implementation (Prometheus, StatsD, etc.).
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
    implementation and the application layer interface. Delegates all image
    processing operations to the wrapped ImageProcessor instance.

    Attributes:
        _processor: The underlying ImageProcessor instance. All operations
            delegate to this processor.

    Note:
        This adapter implements ImageProcessorInterface, allowing the application
        layer to use any image processing implementation without coupling.
    """

    def __init__(self, processor: ImageProcessor) -> None:
        """Initialize the adapter.

        Args:
            processor: ImageProcessor instance to wrap. This processor handles
                all image validation, resizing, compression, and format conversion.
        """
        self._processor = processor

    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Validate and parse image data URL.

        Delegates to underlying processor for data URL validation and parsing.

        Args:
            data_url: Base64-encoded data URL. Format:
                "data:image/{format};base64,{base64_data}"

        Returns:
            Tuple of (format, image_bytes) where:
                - format: Image format string (e.g., "jpeg", "png", "webp")
                - image_bytes: Raw image bytes decoded from base64

        Raises:
            ValueError: If data URL format is invalid, base64 decoding fails,
                or image size exceeds limits.

        Note:
            This method performs validation only. Use process_image() for
            actual image processing and optimization.
        """
        return self._processor.validate_data_url(data_url)

    def process_image(
        self,
        data_url: str,
        target_format: ImageFormat = "jpeg",
    ) -> tuple[str, Any]:  # Returns (base64_string, ImageMetadata)
        """Process and optimize image for VLM model.

        Delegates to underlying processor for image processing, resizing,
        compression, and format conversion.

        Args:
            data_url: Base64-encoded image data URL. Must be valid format.
            target_format: Target image format. One of "jpeg", "png", or "webp".
                Default: "jpeg" (best compression for photos).

        Returns:
            Tuple of (base64_string, metadata) where:
                - base64_string: Processed image as base64 data URL
                - metadata: ImageMetadata object with processing results

        Raises:
            ValueError: If data URL is invalid, image cannot be processed,
                or target format is unsupported.

        Note:
            Processing includes validation, resizing to fit dimension limits,
            compression, and format conversion. The original aspect ratio
            is preserved during resizing.
        """
        return self._processor.process_image(data_url, target_format=target_format)


class ImageCacheAdapter:
    """Adapter that wraps ImageCache to implement ImageCacheInterface.

    This adapter bridges the gap between the infrastructure image cache
    implementation and the application layer interface. Delegates all cache
    operations to the wrapped ImageCache instance.

    Attributes:
        _cache: The underlying ImageCache instance. All operations delegate
            to this cache.

    Note:
        This adapter implements ImageCacheInterface, allowing the application
        layer to use any cache implementation without coupling.
    """

    def __init__(self, cache: ImageCache) -> None:
        """Initialize the adapter.

        Args:
            cache: ImageCache instance to wrap. This cache handles all image
                caching operations (get, put, statistics).
        """
        self._cache = cache

    def get(
        self,
        data_url: str,
        target_format: ImageFormat,
    ) -> tuple[str, Any] | None:  # Returns (base64_string, ImageMetadata) | None
        """Get cached processed image.

        Delegates to underlying cache for retrieval.

        Args:
            data_url: Original image data URL used as cache key component.
            target_format: Target format used as cache key component.

        Returns:
            Tuple of (base64_string, metadata) if cached entry exists and
            is valid, None if cache miss or entry expired.

        Note:
            Cache keys are computed from data_url and target_format combination.
            Expired entries are treated as cache misses.
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

        Delegates to underlying cache for storage.

        Args:
            data_url: Original image data URL used as cache key component.
            target_format: Target format used as cache key component.
            base64_string: Processed image as base64 data URL to cache.
            metadata: ImageMetadata object to cache alongside image.

        Note:
            Cache entries are stored with TTL expiration. If cache is full,
            LRU entries are evicted to make room.
        """
        self._cache.put(data_url, target_format, base64_string, metadata)

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Delegates to underlying cache for statistics retrieval.

        Returns:
            Dictionary with cache statistics containing:
                - size: Current number of cached entries (int)
                - max_size: Maximum cache size (int)
                - hits: Total cache hits (int)
                - misses: Total cache misses (int)
                - hit_rate: Cache hit rate as float (0.0-1.0)

        Note:
            Statistics are cumulative since cache initialization or last clear().
        """
        return self._cache.get_stats()


class AnalyticsCollectorAdapter:
    """Adapter that wraps AnalyticsCollector to implement AnalyticsCollectorInterface.

    This adapter bridges the gap between the infrastructure analytics collection
    implementation and the application layer interface. Delegates to the global
    AnalyticsCollector class.

    Attributes:
        None. Uses the global AnalyticsCollector class from telemetry.analytics.

    Note:
        This is a static adapter - no instance state is maintained. All
        analytics collection is delegated to the global AnalyticsCollector class.
    """

    @staticmethod
    def record_request_with_project(
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        project: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a request metric with project tracking.

        Delegates to the global AnalyticsCollector for project-based analytics.

        Args:
            model: Model name identifier. Use "system" for non-model operations.
            operation: Operation name identifier. Examples: "generate", "chat",
                "vlm", "list_models".
            latency_ms: Request latency in milliseconds. Measured from request
                start to completion.
            success: Whether the request succeeded. True for successful requests,
                False for errors.
            project: Project name from X-Project-Name header. None if not provided.
                Used for project-based analytics and usage tracking.
            error: Error type name if request failed. None if succeeded.
                Examples: "ValueError", "ConnectionError", "TimeoutError".

        Note:
            This method delegates to AnalyticsCollector.record_request_with_project
            which handles the actual analytics collection implementation.
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

    This adapter bridges the gap between the infrastructure performance metrics
    collection implementation and the application layer interface. Delegates to
    the global PerformanceCollector class.

    Attributes:
        None. Uses the global PerformanceCollector class from telemetry.performance.

    Note:
        This is a static adapter - no instance state is maintained. All
        performance collection is delegated to the global PerformanceCollector class.
    """

    @staticmethod
    def record_performance(
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: Any = None,  # GenerateResponse | dict[str, Any] | None
    ) -> None:
        """Record detailed performance metrics.

        Delegates to the global PerformanceCollector for detailed performance tracking.

        Args:
            model: Model name identifier used for the request.
            operation: Operation type identifier. Examples: "generate", "chat",
                "vlm".
            total_latency_ms: Total request latency in milliseconds. Measured
                from request start to completion (including model load time).
            success: Whether the request succeeded. True for successful requests,
                False for errors.
            response: GenerateResponse object or dictionary with detailed timing
                data. Should contain keys like prompt_eval_count, eval_count,
                total_duration, load_duration. None if response data unavailable
                (e.g., on error).

        Note:
            This method delegates to PerformanceCollector.record_performance which
            handles the actual performance collection implementation and calculates
            tokens/second from response data.
        """
        PerformanceCollector.record_performance(
            model=model,
            operation=operation,
            total_latency_ms=total_latency_ms,
            success=success,
            response=response,
        )
