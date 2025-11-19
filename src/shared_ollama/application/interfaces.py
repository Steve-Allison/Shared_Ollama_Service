"""Interfaces (Protocols) for application layer dependencies.

These protocols define the contracts that infrastructure implementations
must satisfy. The application layer depends on these interfaces, not
concrete implementations, enabling dependency inversion.

All interfaces use Python 3.13+ Protocol for structural typing.
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


class ImageProcessorInterface(Protocol):
    """Protocol for image processing implementations.

    Defines the interface for image validation, compression, and conversion.
    Allows different image processing backends without coupling to specific
    implementations.
    """

    def validate_data_url(self, data_url: str) -> tuple[str, bytes]:
        """Validate and parse image data URL.

        Args:
            data_url: Base64-encoded data URL.

        Returns:
            Tuple of (format, image_bytes).

        Raises:
            ValueError: If data URL is invalid.
        """
        ...

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
        ...


class ImageCacheInterface(Protocol):
    """Protocol for image cache implementations.

    Defines the interface for caching processed images. Allows different
    cache backends (in-memory, Redis, etc.) without coupling to specific
    implementations.
    """

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
        ...

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
        ...

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (size, hits, misses, hit_rate, etc.).
        """
        ...


class AnalyticsCollectorInterface(Protocol):
    """Protocol for analytics collection implementations.

    Defines the interface for project-based analytics tracking. Allows
    different analytics backends without coupling to specific implementations.
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

        Args:
            model: Model name or "system" for non-model operations.
            operation: Operation name (e.g., "generate", "chat", "vlm").
            latency_ms: Request latency in milliseconds.
            success: Whether the request succeeded.
            project: Project name from X-Project-Name header. Optional.
            error: Error name if request failed, None if succeeded.
        """
        ...


class PerformanceCollectorInterface(Protocol):
    """Protocol for performance metrics collection implementations.

    Defines the interface for detailed performance metrics tracking. Allows
    different performance backends without coupling to specific implementations.
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

        Args:
            model: Model name used for the request.
            operation: Operation type (e.g., "generate", "chat", "vlm").
            total_latency_ms: Total request latency in milliseconds.
            success: Whether the request succeeded.
            response: GenerateResponse object or dictionary with timing data. Optional.
        """
        ...
