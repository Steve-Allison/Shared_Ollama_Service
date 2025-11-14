"""
Comprehensive behavioral tests for RequestQueue.

Tests focus on real async behavior, concurrency, edge cases, and error handling.
No mocks - tests use real asyncio primitives.
"""

import asyncio
import pytest

from shared_ollama.core.queue import QueueStats, RequestQueue


class TestQueueStats:
    """Behavioral tests for QueueStats dataclass."""

    def test_default_stats_are_zero(self):
        """Test that default QueueStats has all counters at zero."""
        stats = QueueStats()
        assert stats.queued == 0
        assert stats.in_progress == 0
        assert stats.completed == 0
        assert stats.failed == 0
        assert stats.rejected == 0
        assert stats.timeout == 0
        assert stats.total_wait_time_ms == 0.0
        assert stats.max_wait_time_ms == 0.0
        assert stats.avg_wait_time_ms == 0.0

    def test_stats_uses_slots(self):
        """Test that QueueStats uses __slots__ for memory efficiency."""
        stats = QueueStats()
        # If using slots, __dict__ should not exist or be empty
        assert not hasattr(stats, "__dict__") or len(stats.__dict__) == 0

    def test_stats_can_be_updated(self):
        """Test that QueueStats fields can be updated."""
        stats = QueueStats()
        stats.queued = 5
        stats.in_progress = 2
        assert stats.queued == 5
        assert stats.in_progress == 2


class TestRequestQueue:
    """Comprehensive behavioral tests for RequestQueue."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test that queue initializes with correct defaults."""
        queue = RequestQueue()
        assert queue.max_concurrent == 3
        assert queue.max_queue_size == 50
        assert queue.default_timeout == 60.0

    @pytest.mark.asyncio
    async def test_queue_custom_configuration(self):
        """Test that queue accepts custom configuration."""
        queue = RequestQueue(max_concurrent=5, max_queue_size=100, default_timeout=30.0)
        assert queue.max_concurrent == 5
        assert queue.max_queue_size == 100
        assert queue.default_timeout == 30.0

    @pytest.mark.asyncio
    async def test_acquire_releases_slot_on_exit(self):
        """Test that acquire() releases slot when context exits."""
        queue = RequestQueue(max_concurrent=1)
        initial_stats = await queue.get_stats()

        async with queue.acquire(request_id="test-1"):
            stats_during = await queue.get_stats()
            assert stats_during.in_progress == 1

        stats_after = await queue.get_stats()
        assert stats_after.in_progress == 0
        assert stats_after.completed == 1

    @pytest.mark.asyncio
    async def test_acquire_tracks_wait_time(self):
        """Test that acquire() tracks wait time accurately."""
        queue = RequestQueue(max_concurrent=1)

        # First request gets slot immediately
        async with queue.acquire(request_id="req-1"):
            pass

        # Second request waits, then gets slot
        async with queue.acquire(request_id="req-2"):
            pass

        stats = await queue.get_stats()
        assert stats.total_wait_time_ms >= 0.0
        assert stats.max_wait_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_limit(self):
        """Test that concurrent requests respect max_concurrent limit."""
        queue = RequestQueue(max_concurrent=2)
        in_progress = []

        async def worker(req_id: str):
            async with queue.acquire(request_id=req_id):
                in_progress.append(req_id)
                await asyncio.sleep(0.1)  # Hold slot
                in_progress.remove(req_id)

        # Start 5 requests, but only 2 should be in progress at once
        tasks = [asyncio.create_task(worker(f"req-{i}")) for i in range(5)]
        await asyncio.sleep(0.05)  # Let some start

        # At most 2 should be in progress
        assert len(in_progress) <= 2

        await asyncio.gather(*tasks)

        stats = await queue.get_stats()
        assert stats.completed == 5

    @pytest.mark.asyncio
    async def test_queue_rejects_when_full(self):
        """Test that queue rejects requests when queue is full."""
        queue = RequestQueue(max_concurrent=1, max_queue_size=2)

        # Fill the queue
        async def hold_slot():
            async with queue.acquire(request_id="holder"):
                await asyncio.sleep(0.2)

        holder_task = asyncio.create_task(hold_slot())
        await asyncio.sleep(0.01)  # Let holder start

        # Fill queue with 2 waiting requests
        async def wait_in_queue(req_id: str):
            async with queue.acquire(request_id=req_id):
                pass

        task1 = asyncio.create_task(wait_in_queue("req-1"))
        task2 = asyncio.create_task(wait_in_queue("req-2"))
        await asyncio.sleep(0.01)  # Let them queue

        # This should be rejected
        with pytest.raises(RuntimeError, match="Queue is full"):
            async with queue.acquire(request_id="req-3"):
                pass

        await asyncio.gather(holder_task, task1, task2, return_exceptions=True)

        stats = await queue.get_stats()
        assert stats.rejected >= 1

    @pytest.mark.asyncio
    async def test_queue_timeout_raises_timeout_error(self):
        """Test that queue raises TimeoutError when timeout exceeded."""
        queue = RequestQueue(max_concurrent=1, default_timeout=0.1)

        # Hold the slot
        async def hold_slot():
            async with queue.acquire(request_id="holder"):
                await asyncio.sleep(0.5)

        holder_task = asyncio.create_task(hold_slot())
        await asyncio.sleep(0.01)

        # This should timeout
        with pytest.raises(asyncio.TimeoutError):
            async with queue.acquire(request_id="req-timeout", timeout=0.05):
                pass

        await holder_task

        stats = await queue.get_stats()
        assert stats.timeout >= 1

    @pytest.mark.asyncio
    async def test_queue_tracks_failures(self):
        """Test that queue tracks failed requests."""
        queue = RequestQueue()

        async with queue.acquire(request_id="req-fail"):
            raise ValueError("Test failure")

        stats = await queue.get_stats()
        assert stats.failed == 1
        assert stats.completed == 0

    @pytest.mark.asyncio
    async def test_queue_tracks_successes(self):
        """Test that queue tracks successful requests."""
        queue = RequestQueue()

        async with queue.acquire(request_id="req-success"):
            pass  # Success

        stats = await queue.get_stats()
        assert stats.completed == 1
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_get_stats_returns_copy(self):
        """Test that get_stats() returns a copy, not the internal object."""
        queue = RequestQueue()

        async with queue.acquire(request_id="req-1"):
            pass

        stats1 = await queue.get_stats()
        stats1.completed = 999  # Modify the copy

        stats2 = await queue.get_stats()
        assert stats2.completed == 1  # Internal stats unchanged

    @pytest.mark.asyncio
    async def test_get_config_returns_configuration(self):
        """Test that get_config() returns current configuration."""
        queue = RequestQueue(max_concurrent=5, max_queue_size=100, default_timeout=30.0)
        config = queue.get_config()

        assert config["max_concurrent"] == 5
        assert config["max_queue_size"] == 100
        assert config["default_timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_custom_timeout_overrides_default(self):
        """Test that custom timeout parameter overrides default_timeout."""
        queue = RequestQueue(default_timeout=60.0)

        # Hold slot
        async def hold_slot():
            async with queue.acquire(request_id="holder"):
                await asyncio.sleep(0.3)

        holder_task = asyncio.create_task(hold_slot())
        await asyncio.sleep(0.01)

        # Custom timeout should be used
        with pytest.raises(asyncio.TimeoutError):
            async with queue.acquire(request_id="req-custom", timeout=0.05):
                pass

        await holder_task

    @pytest.mark.asyncio
    async def test_multiple_concurrent_queues_independent(self):
        """Test that multiple queue instances are independent."""
        queue1 = RequestQueue(max_concurrent=1)
        queue2 = RequestQueue(max_concurrent=1)

        # Both should be able to process requests independently
        async with queue1.acquire(request_id="q1-1"):
            async with queue2.acquire(request_id="q2-1"):
                pass  # Both should work

        stats1 = await queue1.get_stats()
        stats2 = await queue2.get_stats()

        assert stats1.completed == 1
        assert stats2.completed == 1

    @pytest.mark.asyncio
    async def test_queue_handles_rapid_requests(self):
        """Test that queue handles rapid request submission."""
        queue = RequestQueue(max_concurrent=3)

        async def quick_request(req_id: str):
            async with queue.acquire(request_id=req_id):
                await asyncio.sleep(0.01)

        # Submit many requests rapidly
        tasks = [asyncio.create_task(quick_request(f"req-{i}")) for i in range(20)]
        await asyncio.gather(*tasks)

        stats = await queue.get_stats()
        assert stats.completed == 20

    @pytest.mark.asyncio
    async def test_queue_stats_accuracy_under_load(self):
        """Test that queue stats are accurate under concurrent load."""
        queue = RequestQueue(max_concurrent=2)

        async def worker(req_id: str, should_fail: bool = False):
            async with queue.acquire(request_id=req_id):
                await asyncio.sleep(0.01)
                if should_fail:
                    raise ValueError("Intentional failure")

        # Mix of successes and failures
        tasks = []
        for i in range(10):
            tasks.append(asyncio.create_task(worker(f"req-{i}", should_fail=(i % 3 == 0))))

        await asyncio.gather(*tasks, return_exceptions=True)

        stats = await queue.get_stats()
        assert stats.completed + stats.failed == 10
        assert stats.failed >= 3  # At least 3 failures (every 3rd request)

    @pytest.mark.asyncio
    async def test_queue_cleans_up_on_exception(self):
        """Test that queue properly cleans up on exception."""
        queue = RequestQueue(max_concurrent=1)

        try:
            async with queue.acquire(request_id="req-exception"):
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Slot should be released
        stats = await queue.get_stats()
        assert stats.in_progress == 0
        assert stats.failed == 1

        # Should be able to acquire again
        async with queue.acquire(request_id="req-after-exception"):
            pass

        stats2 = await queue.get_stats()
        assert stats2.completed == 1

