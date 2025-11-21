"""Interfaces (Protocols) for application layer dependencies.

This module defines Protocol-based interfaces that infrastructure
implementations must satisfy. The application layer depends on these
interfaces, not concrete implementations, enabling dependency inversion
and testability.

Design Principles:
    - Structural Typing: Uses Python Protocol for duck typing
    - Dependency Inversion: Application depends on abstractions
    - Interface Segregation: Focused, single-purpose protocols
    - Testability: Easy to mock for unit testing

Key Interfaces:
    - OllamaClientInterface: Client for Ollama service operations
    - RequestLoggerInterface: Structured request logging
    - MetricsCollectorInterface: Basic metrics collection
    - ImageProcessorInterface: Image validation and processing
    - ImageCacheInterface: Image caching operations
    - AnalyticsCollectorInterface: Project-based analytics
    - PerformanceCollectorInterface: Detailed performance metrics

Note:
    All interfaces use Python 3.13+ Protocol for structural typing.
    Implementations don't need to explicitly inherit from these protocols;
    they just need to implement the required methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from shared_ollama.client.sync import GenerateResponse

# Type alias for image formats
ImageFormat = Literal["jpeg", "png", "webp"]


class OllamaClientInterface(Protocol):
    """Protocol for Ollama client implementations.

    Defines the interface that all Ollama client implementations must satisfy.
    This allows the application layer to work with any client implementation
    (async, sync adapter, mock, etc.) without depending on concrete classes.

    Implementations must provide:
        - Model listing (list_models)
        - Text generation (generate)
        - Chat completion (chat)
        - Health checking (health_check)

    All methods are async and may raise ConnectionError for service
    unavailability or other exceptions for infrastructure errors.
    """

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from Ollama service.

        Retrieves the list of all models available in the Ollama service,
        including metadata like size and modification timestamps.

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
            This method should handle service unavailability gracefully
            by raising ConnectionError, allowing callers to distinguish
            service issues from other errors.
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

        Performs single-prompt text generation with optional system message,
        generation options, format constraints, and tool calling capabilities.

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
            format: Output format specification. Can be:
                - "json": Forces JSON output (JSON mode)
                - dict: JSON schema for structured output
                - None: Default text output (no constraints)
            tools: List of tool definitions for function calling. POML compatible.
                Each tool is a dict with "type" and "function" keys. None means
                no tool calling.

        Returns:
            If stream=False: dict with generation result containing:
                - text: Generated text (str)
                - model: Model name used (str)
                - prompt_eval_count: Prompt tokens evaluated (int)
                - eval_count: Generation tokens produced (int)
                - total_duration: Total generation time in nanoseconds (int)
                - load_duration: Model load time in nanoseconds (int)
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with complete metrics.

        Raises:
            ConnectionError: If Ollama service is unavailable or unreachable.
            ValueError: If prompt is empty or options are invalid.
            Exception: For other infrastructure or network errors.

        Note:
            The stream parameter determines return type. Callers must handle
            both dict and AsyncIterator return types appropriately.
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

        Performs multi-turn conversation completion with message history,
        optional images, generation options, format constraints, and tool
        calling capabilities.

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
            images: List of base64-encoded image data URLs (native Ollama format).
                Images are attached to the last user message. None means no images.
            format: Output format specification. Same format as generate().
            tools: List of tool definitions for function calling. POML compatible.
                None means no tool calling.

        Returns:
            If stream=False: dict with chat result containing:
                - message: dict with "role" and "content" keys
                - model: Model name used (str)
                - prompt_eval_count: Prompt tokens evaluated (int)
                - eval_count: Generation tokens produced (int)
                - total_duration: Total generation time in nanoseconds (int)
                - load_duration: Model load time in nanoseconds (int)
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with complete metrics.

        Raises:
            ConnectionError: If Ollama service is unavailable or unreachable.
            ValueError: If messages are invalid or options are invalid.
            Exception: For other infrastructure or network errors.

        Note:
            Images are passed separately (native Ollama format), not embedded
            in message content. For OpenAI-compatible format, use VLM endpoints.
        """
        ...

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy and reachable.

        Performs a lightweight health check to verify the Ollama service
        is available and responding to requests.

        Returns:
            True if service is healthy and reachable, False otherwise.

        Note:
            This method should be fast and non-blocking. It should not
            raise exceptions for service unavailability, but return False
            instead. Exceptions should only be raised for unexpected errors.
        """
        ...


class RequestLoggerInterface(Protocol):
    """Protocol for request logging implementations.

    Defines the interface for structured request logging. Allows different
    logging implementations (file, stdout, cloud services, etc.) without
    coupling to specific implementations.

    Implementations should log structured data in a format suitable for
    analysis (JSON, JSONL, etc.) and include all relevant request metadata.
    """

    def log_request(self, data: dict[str, Any]) -> None:
        """Log a request event with structured data.

        Records a structured log entry for a request event, including
        success/failure status, timing information, and metadata.

        Args:
            data: Dictionary with request event data. Required keys:
                - event: Event type identifier (e.g., "api_request")
                - status: Request status ("success" or "error")
                - request_id: Unique request identifier for tracing
                - operation: Operation name (e.g., "generate", "chat", "vlm")
            Optional keys may include:
                - model: Model name used
                - client_ip: Client IP address
                - project_name: Project identifier
                - latency_ms: Request latency in milliseconds
                - error_type: Error type name (if status="error")
                - error_message: Error message (if status="error")
                - Additional operation-specific fields

        Note:
            This method should be non-blocking and handle errors gracefully.
            Logging failures should not raise exceptions that affect request
            processing.
        """
        ...


class MetricsCollectorInterface(Protocol):
    """Protocol for metrics collection implementations.

    Defines the interface for collecting and recording basic request metrics.
    Allows different metrics backends (Prometheus, StatsD, custom collectors,
    etc.) without coupling to specific implementations.

    This interface focuses on basic metrics (latency, success/failure rates).
    For detailed performance metrics, see PerformanceCollectorInterface.
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

        Records a single request metric for aggregation and analysis.
        Metrics are typically aggregated by model, operation, and success
        status for dashboards and alerting.

        Args:
            model: Model name identifier. Use "system" for non-model operations
                (e.g., "list_models", health checks).
            operation: Operation name identifier. Examples: "generate", "chat",
                "vlm", "list_models".
            latency_ms: Request latency in milliseconds. Should be measured
                from request start to completion (including model load time).
            success: Whether the request succeeded. True for successful requests,
                False for errors.
            error: Error type name if request failed. None if succeeded.
                Examples: "ValueError", "ConnectionError", "TimeoutError".

        Note:
            This method should be non-blocking and handle errors gracefully.
            Metrics collection failures should not raise exceptions that
            affect request processing.
        """
        ...


class ImageProcessorInterface(Protocol):
    """Protocol for image processing implementations.

    Defines the interface for image validation, compression, resizing, and
    format conversion. Allows different image processing backends (PIL/Pillow,
    OpenCV, etc.) without coupling to specific implementations.

    Implementations should handle:
        - Data URL validation and parsing
        - Image format detection and conversion
        - Image resizing to fit dimension limits
        - Image compression for bandwidth optimization
        - Metadata extraction (dimensions, format, size)
    """

    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Validate and parse image data URL.

        Validates the data URL format and extracts the image format and
        raw bytes for processing.

        Args:
            data_url: Base64-encoded data URL. Format:
                "data:image/{format};base64,{base64_data}"

        Returns:
            Tuple of (format, image_bytes) where:
                - format: Image format string (e.g., "jpeg", "png", "webp")
                - image_bytes: Raw image bytes decoded from base64

        Raises:
            ValueError: If data URL format is invalid, base64 decoding fails,
                or image format is unsupported.

        Note:
            This method performs validation only. Use process_image() for
            actual image processing and optimization.
        """
        ...

    def process_image(
        self,
        data_url: str,
        target_format: ImageFormat = "jpeg",
    ) -> tuple[str, Any]:  # Returns (base64_string, ImageMetadata)
        """Process and optimize image for VLM model.

        Validates, resizes, compresses, and converts an image to the target
        format. Images are optimized to fit within dimension limits and
        compressed for efficient transmission to the VLM model.

        Args:
            data_url: Base64-encoded image data URL. Must be valid format.
            target_format: Target image format. One of "jpeg", "png", or "webp".
                Default: "jpeg" (best compression for photos).

        Returns:
            Tuple of (base64_string, metadata) where:
                - base64_string: Processed image as base64 data URL
                - metadata: ImageMetadata object containing:
                    - format: Final image format
                    - width: Image width in pixels
                    - height: Image height in pixels
                    - size_bytes: Image size in bytes
                    - original_size_bytes: Original size before processing
                    - compression_ratio: Compression ratio (if compressed)

        Raises:
            ValueError: If data URL is invalid, image cannot be processed,
                or target format is unsupported.

        Note:
            Processing may resize images to fit max_dimension limits and
            compress them to reduce bandwidth. The original aspect ratio
            is preserved during resizing.
        """
        ...


class ImageCacheInterface(Protocol):
    """Protocol for image cache implementations.

    Defines the interface for caching processed images to avoid redundant
    processing. Allows different cache backends (in-memory LRU, Redis,
    file-based, etc.) without coupling to specific implementations.

    Implementations should handle:
        - Cache key generation from data_url + target_format
        - Cache storage and retrieval
        - Cache invalidation and eviction
        - Cache statistics tracking
    """

    def get(
        self,
        data_url: str,
        target_format: ImageFormat,
    ) -> tuple[str, Any] | None:  # Returns (base64_string, ImageMetadata) | None
        """Get cached processed image.

        Retrieves a previously processed image from cache if available.
        Cache key is derived from data_url and target_format combination.

        Args:
            data_url: Original image data URL used as cache key component.
            target_format: Target format used as cache key component.

        Returns:
            Tuple of (base64_string, metadata) if cached entry exists and
            is valid, None if cache miss or entry expired/invalid.

        Note:
            Implementations should handle cache expiration and invalidation.
            Return None for cache misses rather than raising exceptions.
        """
        ...

    def put(
        self,
        data_url: str,
        target_format: ImageFormat,
        base64_string: str,
        metadata: Any,  # ImageMetadata
    ) -> None:
        """Cache processed image.

        Stores a processed image in cache for future retrieval. Cache key
        is derived from data_url and target_format combination.

        Args:
            data_url: Original image data URL used as cache key component.
            target_format: Target format used as cache key component.
            base64_string: Processed image as base64 data URL to cache.
            metadata: ImageMetadata object to cache alongside image.

        Note:
            Implementations should handle cache eviction when cache is full.
            This method should be non-blocking and handle storage errors
            gracefully without raising exceptions.
        """
        ...

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns current cache statistics for monitoring and optimization.

        Returns:
            Dictionary with cache statistics. Typical keys include:
                - size: Current number of cached entries (int)
                - hits: Total cache hits (int)
                - misses: Total cache misses (int)
                - hit_rate: Cache hit rate as float (0.0-1.0)
                - evictions: Number of entries evicted (int)
                - Additional implementation-specific stats

        Note:
            Statistics should be cumulative since cache creation/reset.
            Implementations may reset stats periodically or on demand.
        """
        ...


class AnalyticsCollectorInterface(Protocol):
    """Protocol for analytics collection implementations.

    Defines the interface for project-based analytics tracking. Allows
    different analytics backends (custom collectors, cloud services, etc.)
    without coupling to specific implementations.

    This interface extends basic metrics with project-level tracking,
    enabling per-project analytics and usage monitoring.
    """

    def record_request_with_project(
        self,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        project: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a request metric with project tracking.

        Records a request metric with optional project association for
        project-based analytics and usage monitoring.

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
            This method should be non-blocking and handle errors gracefully.
            Analytics collection failures should not raise exceptions that
            affect request processing.
        """
        ...


class PerformanceCollectorInterface(Protocol):
    """Protocol for performance metrics collection implementations.

    Defines the interface for detailed performance metrics tracking including
    token counts, model load times, and throughput calculations. Allows
    different performance backends without coupling to specific implementations.

    This interface focuses on detailed performance data (tokens/second,
    model load times, etc.) beyond basic latency metrics.
    """

    def record_performance(
        self,
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: GenerateResponse | dict[str, Any] | None = None,
    ) -> None:
        """Record detailed performance metrics.

        Records comprehensive performance metrics including token counts,
        model load times, and calculated throughput (tokens/second).

        Args:
            model: Model name identifier used for the request.
            operation: Operation type identifier. Examples: "generate", "chat",
                "vlm".
            total_latency_ms: Total request latency in milliseconds. Measured
                from request start to completion (including model load time).
            success: Whether the request succeeded. True for successful requests,
                False for errors.
            response: GenerateResponse object or dictionary with detailed timing
                data. Should contain keys like:
                - prompt_eval_count: Prompt tokens evaluated (int)
                - eval_count: Generation tokens produced (int)
                - total_duration: Total generation time in nanoseconds (int)
                - load_duration: Model load time in nanoseconds (int)
                None if response data unavailable (e.g., on error).

        Note:
            This method calculates tokens/second from eval_count and timing
            data. It should be non-blocking and handle errors gracefully.
            Performance collection failures should not raise exceptions that
            affect request processing.
        """
        ...
