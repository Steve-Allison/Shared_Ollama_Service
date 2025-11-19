"""Request queue management for graceful handling of concurrent requests.

This module provides an async request queue that prevents immediate failures
when capacity is reached, allowing requests to wait for available slots rather
than failing with 503 errors.

Key behaviors:
    - Uses asyncio.Semaphore for concurrency control
    - Tracks comprehensive statistics (wait times, rejections, timeouts)
    - Implements timeout handling for queue waits
    - Thread-safe statistics updates via asyncio.Lock
    - Automatic cleanup via async context managers

Concurrency:
    - All operations are async and safe for concurrent use
    - Statistics updates are protected by asyncio.Lock
    - Queue operations use asyncio.Queue for thread safety
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueStats:
    """Statistics for the request queue.

    Immutable snapshot of queue metrics at a point in time. All time values
    are in milliseconds.

    Attributes:
        queued: Number of requests currently waiting in queue.
        in_progress: Number of requests currently being processed.
        completed: Total requests successfully completed since startup.
        failed: Total requests that failed since startup.
        rejected: Total requests rejected due to full queue since startup.
        timeout: Total requests that timed out waiting in queue since startup.
        total_wait_time_ms: Cumulative time all requests spent waiting (ms).
        max_wait_time_ms: Maximum wait time observed for any request (ms).
        avg_wait_time_ms: Average wait time per request (ms).
    """

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
    """Async request queue with configurable concurrency and queue limits.

    Manages request flow using a semaphore for concurrency control and an
    asyncio.Queue for waiting requests. Tracks comprehensive statistics for
    monitoring and observability.

    Attributes:
        max_concurrent: Maximum number of requests that can be processed
            simultaneously.
        max_queue_size: Maximum number of requests that can wait in queue.
            When full, new requests are rejected immediately.
        default_timeout: Default timeout in seconds for waiting in queue.
            Individual requests can override this.

    Thread safety:
        All operations are async and safe for concurrent use from multiple
        coroutines. Statistics updates are protected by asyncio.Lock.

    Lifecycle:
        - Initialize with __init__()
        - Use acquire() as async context manager for each request
        - Query stats with get_stats() for monitoring
    """

    __slots__ = (
        "max_concurrent",
        "max_queue_size",
        "default_timeout",
        "_semaphore",
        "_queue",
        "_stats_lock",
        "_stats",
        "_active_requests",
    )

    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_size: int = 50,
        default_timeout: float = 60.0,
    ) -> None:
        """Initialize request queue.

        Args:
            max_concurrent: Maximum concurrent requests. Must be positive.
            max_queue_size: Maximum queue depth. Must be positive.
            default_timeout: Default wait timeout in seconds. Must be positive.
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue(maxsize=max_queue_size)

        self._stats_lock = asyncio.Lock()
        self._stats = QueueStats()
        self._active_requests: set[str] = set()

    @asynccontextmanager
    async def acquire(
        self,
        request_id: str,
        timeout: float | None = None,
    ):
        """Acquire a slot for request processing.

        Async context manager that handles the complete request lifecycle:
        queuing, waiting for slot, processing, and cleanup. Automatically
        updates statistics and handles timeouts.

        Args:
            request_id: Unique identifier for the request. Used for tracking
                and logging.
            timeout: Queue wait timeout in seconds. If None, uses default_timeout.
                If exceeded, raises asyncio.TimeoutError.

        Yields:
            None. The context manager guarantees a processing slot is acquired
            while the context is active.

        Raises:
            RuntimeError: If queue is full (max_queue_size reached).
            asyncio.TimeoutError: If timeout exceeded while waiting for slot.

        Side effects:
            - Updates queue statistics (queued, rejected, timeout, etc.)
            - Adds request_id to active_requests set
            - Releases semaphore slot on context exit
            - Logs request lifecycle events

        Example:
            >>> queue = RequestQueue(max_concurrent=3)
            >>> async with queue.acquire(request_id="req-123"):
            ...     result = await process_request()
        """
        timeout = timeout or self.default_timeout
        enqueue_time = time.perf_counter()
        acquired = False

        try:
            if self._queue.full():
                async with self._stats_lock:
                    self._stats.rejected += 1
                raise RuntimeError(
                    f"Queue is full ({self.max_queue_size} requests already queued)"
                )

            await self._queue.put((request_id, enqueue_time))
            async with self._stats_lock:
                self._stats.queued = self._queue.qsize()

            logger.debug(
                "request_queued: request_id=%s, queue_size=%d, in_progress=%d",
                request_id,
                self._queue.qsize(),
                len(self._active_requests),
            )

            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
                acquired = True
            except asyncio.TimeoutError:
                async with self._stats_lock:
                    self._stats.timeout += 1
                    self._stats.queued = self._queue.qsize()

                logger.warning(
                    "request_timeout: request_id=%s, timeout=%ss", request_id, timeout
                )
                raise

            # Use match/case for cleaner error handling (Python 3.13+)
            try:
                queued_id, queued_time = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                wait_time_ms = (time.perf_counter() - queued_time) * 1000

                async with self._stats_lock:
                    total_completed = self._stats.completed + self._stats.failed
                    self._stats.total_wait_time_ms += wait_time_ms
                    self._stats.max_wait_time_ms = max(self._stats.max_wait_time_ms, wait_time_ms)
                    self._stats.queued = self._queue.qsize()
                    self._stats.in_progress = len(self._active_requests) + 1
                    # Calculate average wait time using match/case for clarity
                    match total_completed:
                        case count if count > 0:
                            self._stats.avg_wait_time_ms = (
                                self._stats.total_wait_time_ms / count
                            )
                        case 0:
                            # First request - average equals current wait time
                            self._stats.avg_wait_time_ms = wait_time_ms

                self._active_requests.add(request_id)

                logger.info(
                    "request_processing: request_id=%s, wait_time_ms=%.2f, in_progress=%d",
                    request_id,
                    wait_time_ms,
                    len(self._active_requests),
                )

            except asyncio.TimeoutError:
                logger.error("timeout_getting_from_queue: request_id=%s", request_id)
                async with self._stats_lock:
                    self._stats.queued = self._queue.qsize()
                    self._stats.in_progress = len(self._active_requests) + 1
                self._active_requests.add(request_id)

            yield

            async with self._stats_lock:
                self._stats.completed += 1
                self._stats.in_progress = len(self._active_requests) - 1

            logger.debug("request_completed: request_id=%s", request_id)

        except Exception as exc:
            async with self._stats_lock:
                if request_id in self._active_requests:
                    self._stats.failed += 1
                    self._stats.in_progress = len(self._active_requests) - 1

            # Log differently for exception groups vs regular exceptions
            if isinstance(exc, ExceptionGroup):
                logger.error(
                    "request_failed_exception_group: request_id=%s, exceptions=%s",
                    request_id,
                    [str(e) for e in exc.exceptions],
                )
            else:
                logger.error("request_failed: request_id=%s, error=%s", request_id, exc)
            raise

        finally:
            self._active_requests.discard(request_id)

            if acquired:
                self._semaphore.release()

            async with self._stats_lock:
                self._stats.in_progress = len(self._active_requests)
                self._stats.queued = self._queue.qsize()

    async def get_stats(self) -> QueueStats:
        """Get current queue statistics.

        Returns a snapshot of queue metrics. The returned object is a copy,
        so modifications won't affect internal state.

        Returns:
            QueueStats object with current metrics. All counters reflect
            state since queue initialization.

        Side effects:
            Acquires asyncio.Lock briefly to read statistics atomically.
        """
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
        """Get queue configuration.

        Returns:
            Dictionary with keys:
                - max_concurrent: int
                - max_queue_size: int
                - default_timeout: float
        """
        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "default_timeout": self.default_timeout,
        }


__all__ = ["RequestQueue", "QueueStats"]
