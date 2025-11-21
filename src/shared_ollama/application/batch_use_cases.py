"""Use cases for batch processing operations.

This module defines application use cases for batch processing of multiple
requests concurrently. Batch use cases manage concurrency limits, error
handling, and aggregate results for efficient bulk processing.

Key Features:
    - Concurrent processing with configurable limits
    - Individual request error isolation
    - Aggregate statistics and results
    - Request ID generation for tracing
    - Performance tracking

Design Principles:
    - Concurrency Control: Semaphore limits simultaneous processing
    - Error Isolation: Individual request failures don't stop batch
    - Observability: Comprehensive batch statistics
    - Reusability: Delegates to individual use cases
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared_ollama.application.use_cases import ChatUseCase
    from shared_ollama.application.vlm_use_cases import VLMUseCase
    from shared_ollama.domain.entities import ChatRequest, VLMRequest


class BatchChatUseCase:
    """Use case for batch text-only chat processing.

    Processes multiple chat requests concurrently with configurable concurrency
    limits. Individual request failures are isolated and don't stop the batch.
    Returns aggregate results with success/failure counts and timing.

    Attributes:
        _chat_use_case: Chat use case for processing individual requests.
        _semaphore: Semaphore controlling maximum concurrent requests.

    Note:
        This use case delegates to ChatUseCase for individual request processing.
        Concurrency is controlled via asyncio.Semaphore to prevent resource
        exhaustion. All requests in a batch share the same client_ip and
        project_name for tracking.
    """

    def __init__(
        self,
        chat_use_case: ChatUseCase,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize batch chat use case.

        Args:
            chat_use_case: Chat use case for processing individual requests.
                Used for all requests in the batch.
            max_concurrent: Maximum number of concurrent requests in batch.
                Must be positive. Default: 5. Higher values increase throughput
                but consume more resources.
        """
        self._chat_use_case = chat_use_case
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single(
        self,
        request: ChatRequest,
        batch_id: str,
        index: int,
        client_ip: str | None,
        project_name: str | None,
    ) -> dict[str, Any]:
        """Process a single chat request with semaphore control.

        Processes one chat request from the batch, acquiring semaphore slot
        and handling errors gracefully. Errors are captured and returned in
        the result dict rather than raising exceptions.

        Args:
            request: Chat request domain entity to process.
            batch_id: Unique batch identifier for request ID generation.
            index: Request index within batch (0-based) for ordering.
            client_ip: Client IP address for logging. Shared across batch.
            project_name: Project name for tracking. Shared across batch.

        Returns:
            Dictionary with request result containing:
                - index: Request index in batch (int)
                - success: Whether request succeeded (bool)
                - result: Chat result dict if success=True
                - error: Error message if success=False
                - error_type: Error type name if success=False

        Note:
            This method never raises exceptions. All errors are caught and
            returned in the result dict for batch-level error handling.
        """
        request_id = f"{batch_id}-{index}"
        async with self._semaphore:
            try:
                result = await self._chat_use_case.execute(
                    request=request,
                    request_id=request_id,
                    client_ip=client_ip,
                    project_name=project_name,
                    stream=False,
                )
                return {
                    "index": index,
                    "success": True,
                    "result": result,
                }
            except Exception as exc:
                return {
                    "index": index,
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }

    async def execute(
        self,
        requests: list[ChatRequest],
        client_ip: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Execute batch chat requests.

        Processes multiple chat requests concurrently with concurrency limits.
        All requests are processed in parallel (up to max_concurrent), with
        individual errors isolated and returned in results.

        Args:
            requests: List of chat request domain entities to process. Must not
                be empty. Each request is processed independently.
            client_ip: Client IP address for logging. Shared across all requests
                in batch. None if unavailable.
            project_name: Project name for tracking. Shared across all requests
                in batch. None if not provided.

        Returns:
            Dictionary with batch results containing:
                - batch_id: Unique batch identifier (str)
                - total_requests: Total number of requests in batch (int)
                - successful: Number of successful requests (int)
                - failed: Number of failed requests (int)
                - total_time_ms: Total batch processing time in milliseconds (float)
                - results: List of individual request results, each containing:
                    - index: Request index (int)
                    - success: Success status (bool)
                    - result: Chat result if success=True
                    - error: Error message if success=False
                    - error_type: Error type if success=False

        Note:
            Requests are processed concurrently up to max_concurrent limit.
            Individual request failures don't stop the batch. Results maintain
            original request order via index field.
        """
        batch_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        tasks = [
            self._process_single(req, batch_id, i, client_ip, project_name)
            for i, req in enumerate(requests)
        ]

        results = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - start_time) * 1000

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        return {
            "batch_id": batch_id,
            "total_requests": len(requests),
            "successful": successful,
            "failed": failed,
            "total_time_ms": round(total_time_ms, 3),
            "results": results,
        }


class BatchVLMUseCase:
    """Use case for batch VLM processing.

    Processes multiple VLM requests concurrently with configurable concurrency
    limits. Individual request failures are isolated and don't stop the batch.
    Returns aggregate results with success/failure counts and timing.

    VLM requests are more resource-intensive than text-only chat due to image
    processing, so default concurrency is lower (3 vs 5 for chat).

    Attributes:
        _vlm_use_case: VLM use case for processing individual requests.
        _semaphore: Semaphore controlling maximum concurrent requests.

    Note:
        This use case delegates to VLMUseCase for individual request processing.
        Concurrency is controlled via asyncio.Semaphore to prevent resource
        exhaustion. All requests in a batch share the same client_ip and
        project_name for tracking.
    """

    def __init__(
        self,
        vlm_use_case: VLMUseCase,
        max_concurrent: int = 3,  # Lower for VLM (more resource-intensive)
    ) -> None:
        """Initialize batch VLM use case.

        Args:
            vlm_use_case: VLM use case for processing individual requests.
                Used for all requests in the batch.
            max_concurrent: Maximum number of concurrent requests in batch.
                Must be positive. Default: 3 (lower than chat due to image
                processing overhead). Higher values increase throughput but
                consume more resources (CPU, memory, bandwidth).
        """
        self._vlm_use_case = vlm_use_case
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single(
        self,
        request: VLMRequest,
        batch_id: str,
        index: int,
        client_ip: str | None,
        project_name: str | None,
        target_format: str,
    ) -> dict[str, Any]:
        """Process a single VLM request with semaphore control.

        Processes one VLM request from the batch, acquiring semaphore slot
        and handling errors gracefully. Errors are captured and returned in
        the result dict rather than raising exceptions.

        Args:
            request: VLM request domain entity to process.
            batch_id: Unique batch identifier for request ID generation.
            index: Request index within batch (0-based) for ordering.
            client_ip: Client IP address for logging. Shared across batch.
            project_name: Project name for tracking. Shared across batch.
            target_format: Target image format for processing. One of "jpeg",
                "png", or "webp". Shared across batch.

        Returns:
            Dictionary with request result containing:
                - index: Request index in batch (int)
                - success: Whether request succeeded (bool)
                - result: VLM result dict if success=True
                - error: Error message if success=False
                - error_type: Error type name if success=False

        Note:
            This method never raises exceptions. All errors are caught and
            returned in the result dict for batch-level error handling.
        """
        request_id = f"{batch_id}-{index}"
        async with self._semaphore:
            try:
                result = await self._vlm_use_case.execute(
                    request=request,
                    request_id=request_id,
                    client_ip=client_ip,
                    project_name=project_name,
                    stream=False,
                    target_format=target_format,  # type: ignore[arg-type]
                )
                return {
                    "index": index,
                    "success": True,
                    "result": result,
                }
            except Exception as exc:
                return {
                    "index": index,
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }

    async def execute(
        self,
        requests: list[VLMRequest],
        client_ip: str | None = None,
        project_name: str | None = None,
        target_format: str = "jpeg",
    ) -> dict[str, Any]:
        """Execute batch VLM requests.

        Processes multiple VLM requests concurrently with concurrency limits.
        All requests are processed in parallel (up to max_concurrent), with
        individual errors isolated and returned in results.

        Args:
            requests: List of VLM request domain entities to process. Must not
                be empty. Each request is processed independently.
            client_ip: Client IP address for logging. Shared across all requests
                in batch. None if unavailable.
            project_name: Project name for tracking. Shared across all requests
                in batch. None if not provided.
            target_format: Target image format for processing. One of "jpeg",
                "png", or "webp". Shared across all requests in batch.
                Default: "jpeg" (best compression for photos).

        Returns:
            Dictionary with batch results containing:
                - batch_id: Unique batch identifier (str)
                - total_requests: Total number of requests in batch (int)
                - successful: Number of successful requests (int)
                - failed: Number of failed requests (int)
                - total_time_ms: Total batch processing time in milliseconds (float)
                - results: List of individual request results, each containing:
                    - index: Request index (int)
                    - success: Success status (bool)
                    - result: VLM result if success=True
                    - error: Error message if success=False
                    - error_type: Error type if success=False

        Note:
            Requests are processed concurrently up to max_concurrent limit.
            Individual request failures don't stop the batch. Results maintain
            original request order via index field. Image processing (compression,
            caching) is handled transparently by VLMUseCase.
        """
        batch_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        tasks = [
            self._process_single(req, batch_id, i, client_ip, project_name, target_format)
            for i, req in enumerate(requests)
        ]

        results = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - start_time) * 1000

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        return {
            "batch_id": batch_id,
            "total_requests": len(requests),
            "successful": successful,
            "failed": failed,
            "total_time_ms": round(total_time_ms, 3),
            "results": results,
        }
