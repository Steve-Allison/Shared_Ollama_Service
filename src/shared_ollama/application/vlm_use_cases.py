"""Use cases for vision-language model operations.

This module defines application use cases for vision-language model (VLM)
chat completion. VLM use cases handle image processing, compression, caching,
and VLM-specific optimizations beyond standard text generation.

Key Features:
    - Image validation, resizing, and format conversion
    - Image compression for bandwidth optimization
    - Image caching to avoid redundant processing
    - Tool calling support (POML compatible)
    - Streaming and non-streaming responses
    - Comprehensive metrics and logging

Design Principles:
    - Dependency Inversion: Depends on interfaces, not implementations
    - Image Optimization: Automatic compression and resizing
    - Caching: Reuses processed images to reduce processing overhead
    - Observability: Records metrics, logs, and performance data
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from shared_ollama.domain.entities import VLMRequest
from shared_ollama.domain.exceptions import InvalidRequestError

if TYPE_CHECKING:
    from shared_ollama.application.interfaces import (
        AnalyticsCollectorInterface,
        ImageCacheInterface,
        ImageProcessorInterface,
        MetricsCollectorInterface,
        OllamaClientInterface,
        PerformanceCollectorInterface,
        RequestLoggerInterface,
    )


class VLMUseCase:
    """Use case for vision-language model chat completion.

    Orchestrates VLM chat completion requests with image processing, compression,
    caching, and VLM-specific optimizations. Handles both streaming and
    non-streaming responses with comprehensive observability.

    This use case extends standard chat completion with:
        - Image validation and processing (resize, compress, convert format)
        - Image caching to avoid redundant processing
        - Compression savings tracking
        - VLM-specific message serialization

    Attributes:
        _client: Ollama client adapter for making VLM requests.
        _logger: Request logger for recording request events.
        _metrics: Metrics collector for tracking performance and usage.
        _image_processor: Image processor for validation, resizing, compression.
        _image_cache: Image cache for reusing processed images.
        _analytics: Analytics collector for project-based tracking. Optional.
        _performance: Performance collector for detailed metrics. Optional.

    Note:
        All dependencies are injected via constructor following dependency
        inversion principle. Image processing and caching are transparent
        to the caller.
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
        image_processor: ImageProcessorInterface,
        image_cache: ImageCacheInterface,
        analytics: AnalyticsCollectorInterface | None = None,
        performance: PerformanceCollectorInterface | None = None,
    ) -> None:
        """Initialize VLM use case.

        Args:
            client: Ollama client adapter for making VLM chat requests.
            logger: Request logger for recording request events.
            metrics: Metrics collector for tracking performance and usage.
            image_processor: Image processor for validation, resizing, compression,
                and format conversion.
            image_cache: Image cache for reusing processed images. Reduces
                redundant processing when same images are used multiple times.
            analytics: Analytics collector for project-based tracking. None if
                analytics collection is disabled.
            performance: Performance collector for detailed performance metrics.
                None if detailed performance tracking is disabled.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics
        self._image_processor = image_processor
        self._image_cache = image_cache
        self._analytics = analytics
        self._performance = performance

    def _serialize_messages(
        self,
        request: VLMRequest,
        image_compression: bool,
        target_format: Literal["jpeg", "png", "webp"],
    ) -> tuple[list[dict[str, Any]], int]:
        """Serialize VLM messages with image processing and caching.

        Converts domain VLMMessage entities to client format, processing images
        through validation, compression, format conversion, and caching. Tracks
        compression savings for observability.

        Processing Flow:
            1. Convert message structure (role, content, tool_calls)
            2. For each image in message:
               - Check cache first (if compression enabled)
               - If cache miss: process image (validate, resize, compress, convert)
               - Store in cache for future use
               - Track compression savings
            3. Build client message dict with processed images

        Args:
            request: VLM request domain entity with messages containing images.
            image_compression: Whether to compress images. True enables processing,
                False uses images as-is (only validates).
            target_format: Target image format for processing. One of "jpeg",
                "png", or "webp". Used for format conversion and cache key.

        Returns:
            Tuple of (messages, total_compression_savings) where:
                - messages: List of message dicts in client format with processed
                  images as base64 data URLs
                - total_compression_savings: Total bytes saved through compression
                  (original_size - compressed_size) across all images

        Note:
            Image processing is transparent to the caller. Caching reduces
            redundant processing when the same images are used multiple times.
            Compression savings are tracked for observability and reporting.
        """
        messages: list[dict[str, Any]] = []
        total_compression_savings = 0
        for msg in request.messages:
            message_dict: dict[str, Any] = {"role": msg.role}
            if msg.content is not None:
                message_dict["content"] = msg.content
            if msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id

            if msg.images:
                processed_images: list[str] = []
                for image_url in msg.images:
                    if image_compression:
                        cached = self._image_cache.get(image_url, target_format)
                        if cached:
                            base64_string, metadata = cached
                        else:
                            (
                                base64_string,
                                metadata,
                            ) = self._image_processor.process_image(
                                image_url,
                                target_format=target_format,
                            )
                            self._image_cache.put(
                                image_url,
                                target_format,
                                base64_string,
                                metadata,
                            )
                            total_compression_savings += (
                                metadata.original_size - metadata.compressed_size
                            )
                        processed_images.append(base64_string)
                    else:
                        _, image_bytes = self._image_processor.validate_data_url(image_url)
                        base64_string = (
                            image_bytes.decode("utf-8")
                            if isinstance(image_bytes, bytes)
                            else image_bytes
                        )
                        processed_images.append(base64_string)
                message_dict["images"] = processed_images

            messages.append(message_dict)

        return messages, total_compression_savings

    @staticmethod
    def _build_options_dict(request: VLMRequest) -> dict[str, Any] | None:
        """Build generation options dict from domain entity.

        Converts domain GenerationOptions to client format, removing None
        values to avoid sending unnecessary parameters.

        Args:
            request: VLM request domain entity containing optional GenerationOptions.

        Returns:
            Dictionary with generation options (temperature, top_p, etc.) if
            options are provided, None otherwise. None values are excluded.

        Note:
            This method filters out None values to keep the options dict clean.
            Only provided options are included in the client request.
        """
        if not request.options:
            return None
        options_dict = {
            "temperature": request.options.temperature,
            "top_p": request.options.top_p,
            "top_k": request.options.top_k,
            "repeat_penalty": request.options.repeat_penalty,
            "num_predict": request.options.max_tokens,
            "seed": request.options.seed,
            "stop": request.options.stop,
        }
        return {k: v for k, v in options_dict.items() if v is not None}

    @staticmethod
    def _build_tools_payload(request: VLMRequest) -> list[dict[str, Any]] | None:
        """Build tools payload from domain entity.

        Converts domain Tool entities to client format for POML-compatible
        function calling.

        Args:
            request: VLM request domain entity containing optional tools.

        Returns:
            List of tool dicts in client format if tools are provided, None
            otherwise. Each tool dict contains type and function definition.

        Note:
            Tools are converted to OpenAI/POML-compatible format with type
            and function keys. Function contains name, description, and
            parameters schema.
        """
        if not request.tools:
            return None
        return [
            {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                },
            }
            for tool in request.tools
        ]

    async def execute(
        self,
        request: VLMRequest,
        request_id: str,
        client_ip: str | None = None,
        project_name: str | None = None,
        stream: bool = False,
        target_format: Literal["jpeg", "png", "webp"] = "jpeg",
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Execute a VLM chat request.

        Orchestrates the complete VLM workflow:
        1. Serializes messages with image processing (validation, resizing,
           compression, format conversion, caching)
        2. Builds options and tools payloads from domain entities
        3. Calls client adapter for VLM inference (streaming or non-streaming)
        4. Records metrics, logs request, and tracks performance
        5. Returns result with compression savings metadata

        Args:
            request: VLM request domain entity. Already validated by domain
                layer (VLMRequest.__post_init__). Contains messages with images.
            request_id: Unique request identifier for tracing and logging.
            client_ip: Client IP address for logging and analytics. None if
                unavailable.
            project_name: Project name for request tracking and analytics.
                None if not provided.
            stream: Whether to stream the response incrementally. True returns
                AsyncIterator, False returns dict.
            target_format: Target image format for processing. One of "jpeg",
                "png", or "webp". Default: "jpeg" (best compression for photos).
                Used for format conversion and cache key.

        Returns:
            If stream=False: dict containing VLM result with keys:
                - message: Assistant message dict with role and content
                - model: Model name used
                - prompt_eval_count: Prompt tokens evaluated
                - eval_count: Generation tokens produced
                - total_duration: Total generation time (nanoseconds)
                - load_duration: Model load time (nanoseconds)
                - compression_savings_bytes: Total bytes saved through compression
                  (only if image_compression enabled)
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with complete metrics.

        Raises:
            TypeError: If stream=True but result is not AsyncIterator, or
                stream=False but result is not dict.
            InvalidRequestError: If request violates business rules (rare,
                as domain entities validate themselves).
            ConnectionError: If Ollama service is unavailable.
            Exception: For other client or infrastructure errors.

        Side Effects:
            - Processes and caches images (if compression enabled)
            - Records metrics via MetricsCollectorInterface
            - Logs request via RequestLoggerInterface (includes cache stats)
            - Records analytics via AnalyticsCollectorInterface (if enabled)
            - Records performance via PerformanceCollectorInterface (if enabled)
        """
        start_time = time.perf_counter()

        try:
            messages, total_compression_savings = self._serialize_messages(
                request, request.image_compression, target_format
            )
            model_str = request.model.value if request.model else None
            options_dict = self._build_options_dict(request)
            tools_list = self._build_tools_payload(request)

            result = await self._client.chat(
                messages=messages,
                model=model_str,
                options=options_dict,
                stream=stream,
                format=request.format,
                tools=tools_list,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log and record metrics
            if stream:
                if not isinstance(result, AsyncIterator):
                    raise TypeError("Expected streaming iterator for VLM chat()")
                return result
            if not isinstance(result, dict):
                raise TypeError("Expected dict response for non-streaming VLM chat()")
            result_dict = result
            model_used = result_dict.get("model", model_str or "unknown")
            load_duration = result_dict.get("load_duration", 0)
            model_load_ms = round(load_duration / 1_000_000, 3) if load_duration else None
            model_warm_start = (load_duration == 0) if load_duration is not None else None

            # Get cache stats
            cache_stats = self._image_cache.get_stats()

            self._logger.log_request(
                {
                    "event": "api_request",
                    "client_type": "rest_api",
                    "operation": "vlm",
                    "status": "success",
                    "model": model_used,
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "project_name": project_name,
                    "latency_ms": round(latency_ms, 3),
                    "model_load_ms": model_load_ms,
                    "model_warm_start": model_warm_start,
                    "images_count": sum(len(msg.get("images", [])) for msg in messages),
                    "image_compression_enabled": request.image_compression,
                    "compression_savings_bytes": total_compression_savings,
                    "cache_hit_rate": cache_stats["hit_rate"],
                }
            )

            self._metrics.record_request(
                model=model_used,
                operation="vlm",
                latency_ms=latency_ms,
                success=True,
            )

            # Record project-based analytics
            if self._analytics:
                self._analytics.record_request_with_project(
                    model=model_used,
                    operation="vlm",
                    latency_ms=latency_ms,
                    success=True,
                    project=project_name,
                )

            # Record detailed performance metrics
            if self._performance:
                self._performance.record_performance(
                    model=model_used,
                    operation="vlm",
                    total_latency_ms=latency_ms,
                    success=True,
                    response=result_dict,
                )

            # Add compression savings to result if compression was enabled
            if request.image_compression and total_compression_savings > 0:
                result_dict["compression_savings_bytes"] = total_compression_savings

            return result_dict

        except ValueError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request(
                {
                    "event": "api_request",
                    "client_type": "rest_api",
                    "operation": "vlm",
                    "status": "error",
                    "model": model_str,
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "project_name": project_name,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "ValueError",
                    "error_message": str(exc),
                }
            )

            self._metrics.record_request(
                model=model_str,
                operation="vlm",
                latency_ms=latency_ms,
                success=False,
                error="ValueError",
            )

            if self._analytics:
                self._analytics.record_request_with_project(
                    model=model_str,
                    operation="vlm",
                    latency_ms=latency_ms,
                    success=False,
                    project=project_name,
                    error="ValueError",
                )

            raise InvalidRequestError(f"Invalid VLM request: {exc!s}") from exc

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request(
                {
                    "event": "api_request",
                    "client_type": "rest_api",
                    "operation": "vlm",
                    "status": "error",
                    "model": model_str,
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "project_name": project_name,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            self._metrics.record_request(
                model=model_str,
                operation="vlm",
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )

            if self._analytics:
                self._analytics.record_request_with_project(
                    model=model_str,
                    operation="vlm",
                    latency_ms=latency_ms,
                    success=False,
                    project=project_name,
                    error=type(exc).__name__,
                )

            raise
