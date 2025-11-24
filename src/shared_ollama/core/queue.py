"""Bounded async request queue with instrumentation."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueStats:
    """Immutable snapshot of queue metrics (timings in milliseconds)."""

    queued: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    timeout: int = 0
    total_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    avg_wait_time_ms: float = 0.0


class RequestQueue:
    """Async queue that enforces concurrency limits and tracks telemetry."""

    __slots__ = (
        "max_concurrent",
        "max_queue_size",
        "default_timeout",
        "_active_requests",
        "_queue",
        "_semaphore",
        "_stats",
        "_stats_lock",
    )

    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_size: int = 50,
        default_timeout: float = 60.0,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue(maxsize=max_queue_size)
        self._stats_lock = asyncio.Lock()
        self._stats = QueueStats()
        self._active_requests: set[str] = set()

    @asynccontextmanager
    async def acquire(self, request_id: str, timeout: float | None = None) -> AsyncIterator[None]:
        """Reserve a processing slot for *request_id*.

        Example
        -------
        >>> queue = RequestQueue(max_concurrent=2)
        >>> async with queue.acquire("req-1"):
        ...     ...
        """

        timeout = timeout or self.default_timeout
        queued_at = time.perf_counter()
        acquired = False

        if self._queue.full():
            await self._register_rejection()
            raise RuntimeError(f"Queue is full ({self.max_queue_size} requests already queued)")

        await self._queue.put((request_id, queued_at))
        await self._refresh_queue_size()
        logger.debug("request_queued id=%s queued=%d", request_id, self._queue.qsize())

        try:
            try:
                async with asyncio.timeout(timeout):
                    await self._semaphore.acquire()
            except TimeoutError:
                await self._register_timeout()
                logger.warning("request_timeout id=%s timeout=%ss", request_id, timeout)
                raise

            acquired = True
            wait_time_ms = await self._drain_queue_entry(request_id)
            self._active_requests.add(request_id)
            await self._register_wait(wait_time_ms)

            logger.info(
                "request_processing id=%s wait_ms=%.2f in_progress=%d",
                request_id,
                wait_time_ms,
                len(self._active_requests),
            )

            yield

            await self._register_completion()
            logger.debug("request_completed id=%s", request_id)

        except Exception as exc:
            await self._register_failure(request_id)
            if isinstance(exc, ExceptionGroup):
                logger.exception(
                    "request_failed_exception_group id=%s exceptions=%s",
                    request_id,
                    [str(e) for e in exc.exceptions],
                )
            else:
                logger.exception("request_failed id=%s", request_id)
            raise

        finally:
            self._active_requests.discard(request_id)
            if acquired:
                self._semaphore.release()
            await self._refresh_usage_counters()

    async def get_stats(self) -> QueueStats:
        """Return a snapshot of queue metrics."""

        async with self._stats_lock:
            return QueueStats(
                queued=self._stats.queued,
                in_progress=self._stats.in_progress,
                completed=self._stats.completed,
                failed=self._stats.failed,
                rejected=self._stats.rejected,
                timeout=self._stats.timeout,
                total_wait_time_ms=self._stats.total_wait_time_ms,
                max_wait_time_ms=self._stats.max_wait_time_ms,
                avg_wait_time_ms=self._stats.avg_wait_time_ms,
            )

    def get_config(self) -> dict[str, Any]:
        """Return queue configuration for diagnostics endpoints."""

        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "default_timeout": self.default_timeout,
        }

    async def _drain_queue_entry(self, request_id: str) -> float:
        try:
            async with asyncio.timeout(1):
                _, enqueued = await self._queue.get()
        except TimeoutError:  # pragma: no cover - defensive
            logger.warning("queue_get_timeout id=%s", request_id)
            enqueued = time.perf_counter()
        return (time.perf_counter() - enqueued) * 1000

    async def _register_wait(self, wait_time_ms: float) -> None:
        async with self._stats_lock:
            completed_or_failed = self._stats.completed + self._stats.failed or 1
            self._stats.total_wait_time_ms += wait_time_ms
            self._stats.max_wait_time_ms = max(self._stats.max_wait_time_ms, wait_time_ms)
            self._stats.avg_wait_time_ms = self._stats.total_wait_time_ms / completed_or_failed
            self._stats.in_progress = len(self._active_requests)
            self._stats.queued = self._queue.qsize()

    async def _register_completion(self) -> None:
        async with self._stats_lock:
            self._stats.completed += 1

    async def _register_failure(self, request_id: str) -> None:
        async with self._stats_lock:
            if request_id in self._active_requests:
                self._stats.failed += 1

    async def _register_rejection(self) -> None:
        async with self._stats_lock:
            self._stats.rejected += 1

    async def _register_timeout(self) -> None:
        async with self._stats_lock:
            self._stats.timeout += 1
            self._stats.queued = self._queue.qsize()

    async def _refresh_queue_size(self) -> None:
        async with self._stats_lock:
            self._stats.queued = self._queue.qsize()

    async def _refresh_usage_counters(self) -> None:
        async with self._stats_lock:
            self._stats.in_progress = len(self._active_requests)
            self._stats.queued = self._queue.qsize()


__all__ = ["QueueStats", "RequestQueue"]
