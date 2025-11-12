"""
Request queue management for graceful handling of concurrent requests.

Provides a queue system that prevents immediate failures when capacity is reached,
allowing requests to wait for available slots rather than failing with 503 errors.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    """Statistics for the request queue."""

    queued: int = 0  # Currently waiting in queue
    in_progress: int = 0  # Currently being processed
    completed: int = 0  # Successfully completed since startup
    failed: int = 0  # Failed since startup
    rejected: int = 0  # Rejected (queue full) since startup
    timeout: int = 0  # Timed out waiting in queue since startup
    total_wait_time_ms: float = 0.0  # Total time requests spent waiting
    max_wait_time_ms: float = 0.0  # Maximum wait time observed
    avg_wait_time_ms: float = 0.0  # Average wait time


class RequestQueue:
    """
    Async request queue with configurable concurrency and queue limits.

    Provides graceful handling of traffic spikes by queueing requests
    instead of immediately rejecting them.

    Usage:
        queue = RequestQueue(max_concurrent=3, max_queue_size=50)

        async with queue.acquire(request_id="req-123", timeout=60.0):
            # Process request
            result = await do_work()

    Context manager automatically:
    - Waits for available slot (or times out)
    - Tracks request in statistics
    - Releases slot when done
    - Handles errors and cleanup
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_size: int = 50,
        default_timeout: float = 60.0,
    ):
        """
        Initialize request queue.

        Args:
            max_concurrent: Maximum number of concurrent requests
            max_queue_size: Maximum number of queued requests
            default_timeout: Default timeout for queue wait (seconds)
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue(maxsize=max_queue_size)

        # Statistics tracking
        self._stats_lock = asyncio.Lock()
        self._stats = QueueStats()
        self._active_requests: set[str] = set()

    @asynccontextmanager
    async def acquire(self, request_id: str, timeout: float | None = None):
        """
        Acquire a slot for request processing.

        Args:
            request_id: Unique identifier for the request
            timeout: Queue wait timeout in seconds (uses default if None)

        Raises:
            asyncio.TimeoutError: If timeout waiting in queue
            RuntimeError: If queue is full

        Yields:
            None - but guarantees slot is acquired while in context
        """
        timeout = timeout or self.default_timeout
        enqueue_time = time.perf_counter()
        acquired = False

        try:
            # Try to add to queue (non-blocking check)
            if self._queue.full():
                async with self._stats_lock:
                    self._stats.rejected += 1
                raise RuntimeError(f"Queue is full ({self.max_queue_size} requests already queued)")

            # Add to queue
            await self._queue.put((request_id, enqueue_time))
            async with self._stats_lock:
                self._stats.queued = self._queue.qsize()

            logger.debug(
                "request_queued: request_id=%s, queue_size=%d, in_progress=%d",
                request_id,
                self._queue.qsize(),
                len(self._active_requests),
            )

            # Wait for semaphore with timeout
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
                acquired = True
            except asyncio.TimeoutError:
                # Remove from queue if we timeout
                try:
                    # Best effort removal - queue may have changed
                    pass
                except Exception:
                    pass

                async with self._stats_lock:
                    self._stats.timeout += 1
                    self._stats.queued = self._queue.qsize()

                logger.warning("request_timeout: request_id=%s, timeout=%ss", request_id, timeout)
                raise

            # Remove from queue (we got the slot)
            try:
                queued_id, queued_time = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                wait_time_ms = (time.perf_counter() - queued_time) * 1000

                async with self._stats_lock:
                    self._stats.queued = self._queue.qsize()
                    self._stats.in_progress = len(self._active_requests) + 1
                    self._stats.total_wait_time_ms += wait_time_ms
                    if wait_time_ms > self._stats.max_wait_time_ms:
                        self._stats.max_wait_time_ms = wait_time_ms

                    total_completed = self._stats.completed + self._stats.failed
                    if total_completed > 0:
                        self._stats.avg_wait_time_ms = (
                            self._stats.total_wait_time_ms / total_completed
                        )

                self._active_requests.add(request_id)

                logger.info(
                    "request_processing: request_id=%s, wait_time_ms=%.2f, in_progress=%d",
                    request_id,
                    wait_time_ms,
                    len(self._active_requests),
                )

            except asyncio.TimeoutError:
                # Shouldn't happen, but handle gracefully
                logger.error("timeout_getting_from_queue: request_id=%s", request_id)
                async with self._stats_lock:
                    self._stats.queued = self._queue.qsize()
                    self._stats.in_progress = len(self._active_requests) + 1
                self._active_requests.add(request_id)

            # Process request (yield to context)
            yield

            # Success - update stats
            async with self._stats_lock:
                self._stats.completed += 1
                self._stats.in_progress = len(self._active_requests) - 1

            logger.debug("request_completed: request_id=%s", request_id)

        except Exception as exc:
            # Failure - update stats
            async with self._stats_lock:
                if request_id in self._active_requests:
                    self._stats.failed += 1
                    self._stats.in_progress = len(self._active_requests) - 1

            logger.error("request_failed: request_id=%s, error=%s", request_id, exc)
            raise

        finally:
            # Always cleanup
            if request_id in self._active_requests:
                self._active_requests.remove(request_id)

            if acquired:
                self._semaphore.release()

            async with self._stats_lock:
                self._stats.in_progress = len(self._active_requests)
                self._stats.queued = self._queue.qsize()

    async def get_stats(self) -> QueueStats:
        """
        Get current queue statistics.

        Returns:
            QueueStats object with current metrics
        """
        async with self._stats_lock:
            # Return a copy to prevent external modification
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
        """
        Get queue configuration.

        Returns:
            Dictionary with configuration values
        """
        return {
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "default_timeout": self.default_timeout,
        }


__all__ = ["RequestQueue", "QueueStats"]
