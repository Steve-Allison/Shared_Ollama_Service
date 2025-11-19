"""Metrics collection and monitoring utilities for the Shared Ollama Service.

This module provides in-memory metrics collection for request tracking,
performance monitoring, and service health observability.

Key behaviors:
    - In-memory storage with automatic size limiting (max 10,000 metrics)
    - Time-window filtering for recent metrics analysis
    - Statistical calculations (percentiles, averages)
    - Thread-safe operations (single-threaded by design)
    - Automatic timestamp tracking with UTC timezone

Memory management:
    - Metrics are stored in a list with automatic trimming
    - Oldest metrics are discarded when limit is reached
    - No external dependencies for storage (in-memory only)
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Self

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RequestMetrics:
    """Metrics for a single request.

    Immutable snapshot of a single request's performance characteristics.
    All timestamps are timezone-aware (UTC).

    Attributes:
        model: Model name used for the request.
        operation: Operation type (e.g., "generate", "chat", "list_models").
        latency_ms: Request latency in milliseconds (>=0.0).
        success: Whether the request succeeded.
        error: Error message or type if request failed. None if successful.
        timestamp: Request timestamp in UTC. Automatically set to current time.
    """

    model: str
    operation: str
    latency_ms: float
    success: bool
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class ServiceMetrics:
    """Aggregated service metrics.

    Contains aggregated statistics computed from RequestMetrics collection.
    All time values are in milliseconds. Timestamps are timezone-aware (UTC).

    Attributes:
        total_requests: Total number of requests in the aggregation window.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        requests_by_model: Dictionary mapping model names to request counts.
        requests_by_operation: Dictionary mapping operation types to request counts.
        average_latency_ms: Average request latency in milliseconds.
        p50_latency_ms: 50th percentile (median) latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        p99_latency_ms: 99th percentile latency in milliseconds.
        errors_by_type: Dictionary mapping error types to occurrence counts.
        last_request_time: Timestamp of most recent request. None if no requests.
        first_request_time: Timestamp of oldest request. None if no requests.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_operation: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    errors_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_request_time: datetime | None = None
    first_request_time: datetime | None = None


class MetricsCollector:
    """Collects and aggregates metrics for the Ollama service.

    Thread-safe class-level metrics storage with automatic size limiting.
    Provides efficient aggregation and statistical calculations.

    Attributes:
        _metrics: Class variable storing list of RequestMetrics.
        _max_metrics: Maximum number of metrics to retain (default: 10,000).

    Thread safety:
        Not thread-safe. Use from a single thread or protect with locks
        if accessing from multiple threads.

    Memory management:
        Automatically trims oldest metrics when _max_metrics is exceeded.
        Uses efficient list slicing for O(n) trimming operation.
    """

    _metrics: ClassVar[list[RequestMetrics]] = []
    _max_metrics: ClassVar[int] = 10_000

    @classmethod
    def record_request(
        cls,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record a request metric.

        Adds a new RequestMetrics entry to the collection. Automatically
        trims oldest metrics if collection exceeds _max_metrics.

        Args:
            model: Model name used for the request.
            operation: Operation type (e.g., "generate", "chat").
            latency_ms: Request latency in milliseconds. Must be >= 0.0.
            success: Whether the request succeeded.
            error: Error message or type if request failed. None if successful.

        Side effects:
            - Appends to _metrics list
            - May trim oldest metrics if limit exceeded
            - Logs debug message
        """
        metric = RequestMetrics(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
            timestamp=datetime.now(UTC),
        )
        cls._metrics.append(metric)

        if len(cls._metrics) > cls._max_metrics:
            cls._metrics = cls._metrics[-cls._max_metrics :]

        logger.debug("Recorded metric: %s on %s - %.2fms", operation, model, latency_ms)

    @classmethod
    def get_metrics(cls, window_minutes: int | None = None) -> ServiceMetrics:
        """Get aggregated metrics for a time window.

        Computes comprehensive statistics from collected metrics, optionally
        filtered by time window. Uses efficient filtering and modern
        statistical calculations.

        Args:
            window_minutes: Time window in minutes. If None, aggregates all
                metrics. Only metrics within the window are included.

        Returns:
            ServiceMetrics with aggregated statistics. Returns empty
            ServiceMetrics if no metrics available or window is empty.

        Side effects:
            - Filters metrics by timestamp if window_minutes specified
            - Sorts latencies for percentile calculation
            - Computes statistics (O(n log n) for sorting)
        """
        if not cls._metrics:
            return ServiceMetrics()

        # Use match/case for window filtering (Python 3.13+)
        match window_minutes:
            case None:
                metrics = cls._metrics
            case minutes if minutes > 0:
                cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
                metrics = [m for m in cls._metrics if m.timestamp >= cutoff]
            case _:
                metrics = []

        # Guard clause: early return if no metrics
        if not metrics:
            return ServiceMetrics()

        # Performance optimization: single pass for latencies and aggregation
        latencies = [m.latency_ms for m in metrics]
        latencies_sorted = sorted(latencies)

        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        failed = total - successful

        # Performance optimization: use Counter for O(n) aggregation
        # More efficient than defaultdict + loop for counting
        from collections import Counter

        requests_by_model = dict(Counter(m.model for m in metrics))
        requests_by_operation = dict(Counter(m.operation for m in metrics))
        # Filter errors and count in single pass
        errors_by_type = dict(Counter(m.error for m in metrics if m.error))

        match len(latencies_sorted):
            case n if n >= 2:
                quantiles = statistics.quantiles(latencies_sorted, n=100)
                p50 = quantiles[49]
                p95 = quantiles[94]
                p99 = quantiles[98]
            case 1:
                p50 = p95 = p99 = latencies_sorted[0]
            case _:
                p50 = p95 = p99 = 0.0

        return ServiceMetrics(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            requests_by_model=dict(requests_by_model),
            requests_by_operation=dict(requests_by_operation),
            average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            errors_by_type=dict(errors_by_type),
            last_request_time=max(m.timestamp for m in metrics),
            first_request_time=min(m.timestamp for m in metrics),
        )

    @classmethod
    def get_metrics_json(cls, window_minutes: int | None = None) -> dict[str, Any]:
        """Get metrics as JSON-serializable dictionary.

        Convenience method that converts ServiceMetrics to a dictionary
        suitable for JSON serialization. Timestamps are converted to ISO
        format strings.

        Args:
            window_minutes: Time window in minutes. If None, aggregates all
                metrics.

        Returns:
            Dictionary with metrics data. All numeric values are rounded to
            2 decimal places. Timestamps are ISO 8601 strings or None.
        """
        metrics = cls.get_metrics(window_minutes)
        return {
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "requests_by_model": metrics.requests_by_model,
            "requests_by_operation": metrics.requests_by_operation,
            "average_latency_ms": round(metrics.average_latency_ms, 2),
            "p50_latency_ms": round(metrics.p50_latency_ms, 2),
            "p95_latency_ms": round(metrics.p95_latency_ms, 2),
            "p99_latency_ms": round(metrics.p99_latency_ms, 2),
            "errors_by_type": metrics.errors_by_type,
            "last_request_time": metrics.last_request_time.isoformat() if metrics.last_request_time else None,
            "first_request_time": metrics.first_request_time.isoformat() if metrics.first_request_time else None,
        }

    @classmethod
    def reset(cls) -> Self:
        """Reset all collected metrics.

        Clears the entire metrics collection. Useful for testing or
        periodic resets to prevent unbounded memory growth.

        Side effects:
            Sets _metrics to empty list.

        Returns:
            Self for method chaining (Python 3.13+ Self type).
        """
        cls._metrics = []
        return cls


@contextmanager
def track_request(model: str, operation: str) -> Generator[None, None, None]:
    """Context manager to track a request's latency.

    Automatically measures execution time and records metrics. Handles
    exceptions by recording error information.

    Args:
        model: Model name for the request.
        operation: Operation type (e.g., "generate", "chat").

    Yields:
        None. The context manager tracks timing while the context is active.

    Side effects:
        - Measures execution time using time.perf_counter()
        - Records metrics via MetricsCollector.record_request()
        - Logs error information if exception occurs

    Example:
        >>> with track_request("qwen2.5vl:7b", "generate"):
        ...     result = await client.generate("Hello")
    """
    start = time.perf_counter()
    error: str | None = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        MetricsCollector.record_request(
            model=model,
            operation=operation,
            latency_ms=duration_ms,
            success=error is None,
            error=error,
        )


def get_metrics_endpoint() -> dict[str, Any]:
    """Convenience helper for exposing metrics via an HTTP endpoint.

    Returns all metrics as a JSON-serializable dictionary. Useful for
    implementing /metrics or /stats endpoints.

    Returns:
        Dictionary with current metrics. Same format as
        MetricsCollector.get_metrics_json().
    """
    return MetricsCollector.get_metrics_json()


__all__ = ["MetricsCollector", "ServiceMetrics", "get_metrics_endpoint", "track_request"]
