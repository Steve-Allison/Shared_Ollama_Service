"""Use cases for batch processing operations."""

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
    """Use case for batch text-only chat processing."""

    def __init__(
        self,
        chat_use_case: ChatUseCase,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize batch chat use case.

        Args:
            chat_use_case: Chat use case for individual requests.
            max_concurrent: Maximum concurrent chat requests in batch.
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
        """Process a single chat request with semaphore control."""
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

        Args:
            requests: List of chat requests to process.
            client_ip: Client IP address.
            project_name: Project name for tracking.

        Returns:
            Dictionary with batch results and metadata.
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
    """Use case for batch VLM processing."""

    def __init__(
        self,
        vlm_use_case: VLMUseCase,
        max_concurrent: int = 3,  # Lower for VLM (more resource-intensive)
    ) -> None:
        """Initialize batch VLM use case.

        Args:
            vlm_use_case: VLM use case for individual requests.
            max_concurrent: Maximum concurrent VLM requests in batch.
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
        """Process a single VLM request with semaphore control."""
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

        Args:
            requests: List of VLM requests to process.
            client_ip: Client IP address.
            project_name: Project name for tracking.
            target_format: Image compression format.

        Returns:
            Dictionary with batch results and metadata.
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
