"""Use cases for vision-language model operations."""

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

    Handles image processing, compression, caching, and VLM-specific optimizations.
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
            client: Ollama client implementation.
            logger: Request logger implementation.
            metrics: Metrics collector implementation.
            image_processor: Image processing utility.
            image_cache: Image cache for reusing compressed images.
            analytics: Analytics collector implementation. Optional.
            performance: Performance collector implementation. Optional.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics
        self._image_processor = image_processor
        self._image_cache = image_cache
        self._analytics = analytics
        self._performance = performance

    def _serialize_messages(self, request: VLMRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
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
            messages.append(message_dict)
        return messages

    def _process_images(
        self,
        request: VLMRequest,
        target_format: Literal["jpeg", "png", "webp"],
    ) -> tuple[list[str], int]:
        all_images: list[str] = []
        total_compression_savings = 0
        for image_url in request.images:
            if request.image_compression:
                cached = self._image_cache.get(image_url, target_format)
                if cached:
                    base64_string, metadata = cached
                else:
                    base64_string, metadata = self._image_processor.process_image(
                        image_url,
                        target_format=target_format,
                    )
                    self._image_cache.put(
                        image_url,
                        target_format,
                        base64_string,
                        metadata,
                    )
                    total_compression_savings += metadata.original_size - metadata.compressed_size
            else:
                _, image_bytes = self._image_processor.validate_data_url(image_url)
                base64_string = image_bytes.decode("utf-8") if isinstance(image_bytes, bytes) else image_bytes
            all_images.append(base64_string)
        return all_images, total_compression_savings

    @staticmethod
    def _build_options_dict(request: VLMRequest) -> dict[str, Any] | None:
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

        Orchestrates VLM workflow:
        1. Validates request
        2. Processes and compresses images (with caching)
        3. Calls client for VLM inference
        4. Logs request and records metrics

        Args:
            request: VLM request domain entity.
            request_id: Unique request identifier.
            client_ip: Client IP address for logging.
            project_name: Project name for logging.
            stream: Whether to stream the response.
            target_format: Image compression format (jpeg or png).

        Returns:
            - dict with VLM result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            InvalidRequestError: If request violates business rules.
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        start_time = time.perf_counter()

        try:
            messages = self._serialize_messages(request)
            all_images, total_compression_savings = self._process_images(request, target_format)
            model_str = request.model.value if request.model else None
            options_dict = self._build_options_dict(request)
            tools_list = self._build_tools_payload(request)

            result = await self._client.chat(
                messages=messages,
                model=model_str,
                options=options_dict,
                stream=stream,
                images=all_images if all_images else None,
                format=request.format,
                tools=tools_list,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log and record metrics
            if stream:
                return result
            model_used = result.get("model", model_str or "unknown")
            load_duration = result.get("load_duration", 0)
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
                    "images_count": len(all_images),
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
                    response=result,
                )

            # Add compression savings to result if compression was enabled
            if request.image_compression and total_compression_savings > 0:
                result["compression_savings_bytes"] = total_compression_savings

            return result

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
