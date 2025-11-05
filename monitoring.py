"""
Monitoring and Metrics for Shared Ollama Service
================================================

Provides usage tracking, metrics collection, and observability for the central
model infrastructure service.

Usage:
    from monitoring import MetricsCollector, track_request

    # Track a request
    with track_request("generate", model="qwen2.5vl:7b"):
        response = client.generate("Hello!")

    # Get metrics
    metrics = MetricsCollector.get_metrics()
    print(f"Total requests: {metrics['total_requests']}")
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    model: str
    operation: str
    latency_ms: float
    success: bool
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ServiceMetrics:
    """Aggregated service metrics."""

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
    """
    Collects and aggregates metrics for the Ollama service.

    This is a simple in-memory metrics collector. For production,
    consider integrating with Prometheus, StatsD, or similar.
    """

    _metrics: ClassVar[list[RequestMetrics]] = []
    _max_metrics: ClassVar[int] = 10000  # Keep last 10k requests

    @classmethod
    def record_request(
        cls,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ):
        """
        Record a request metric.

        Args:
            model: Model name used
            operation: Operation type (generate, chat, etc.)
            latency_ms: Request latency in milliseconds
            success: Whether request was successful
            error: Error message if request failed
        """
        metric = RequestMetrics(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
            timestamp=datetime.now(),
        )
        cls._metrics.append(metric)

        # Prune old metrics if we exceed max
        if len(cls._metrics) > cls._max_metrics:
            cls._metrics = cls._metrics[-cls._max_metrics :]

        logger.debug(f"Recorded metric: {operation} on {model} - {latency_ms:.2f}ms")

    @classmethod
    def get_metrics(cls, window_minutes: int | None = None) -> ServiceMetrics:
        """
        Get aggregated metrics.

        Args:
            window_minutes: Only include metrics from last N minutes (None = all)

        Returns:
            ServiceMetrics with aggregated statistics
        """
        if not cls._metrics:
            return ServiceMetrics()

        # Filter by time window if specified
        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            metrics = [m for m in cls._metrics if m.timestamp >= cutoff]
        else:
            metrics = cls._metrics

        if not metrics:
            return ServiceMetrics()

        # Calculate statistics
        latencies = [m.latency_ms for m in metrics]
        latencies_sorted = sorted(latencies)

        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        failed = total - successful

        requests_by_model = defaultdict(int)
        requests_by_operation = defaultdict(int)
        errors_by_type = defaultdict(int)

        for m in metrics:
            requests_by_model[m.model] += 1
            requests_by_operation[m.operation] += 1
            if m.error:
                errors_by_type[m.error] += 1

        def percentile(data: list[float], p: float) -> float:
            """Calculate percentile."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] + c * (data[f + 1] - data[f])
            return data[f]

        return ServiceMetrics(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            requests_by_model=dict(requests_by_model),
            requests_by_operation=dict(requests_by_operation),
            average_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency_ms=percentile(latencies_sorted, 0.50),
            p95_latency_ms=percentile(latencies_sorted, 0.95),
            p99_latency_ms=percentile(latencies_sorted, 0.99),
            errors_by_type=dict(errors_by_type),
            last_request_time=max(m.timestamp for m in metrics),
            first_request_time=min(m.timestamp for m in metrics),
        )

    @classmethod
    def get_metrics_json(cls, window_minutes: int | None = None) -> dict[str, Any]:
        """
        Get metrics as JSON-serializable dictionary.

        Args:
            window_minutes: Only include metrics from last N minutes (None = all)

        Returns:
            Dictionary with metrics
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
            "last_request_time": (
                metrics.last_request_time.isoformat() if metrics.last_request_time else None
            ),
            "first_request_time": (
                metrics.first_request_time.isoformat() if metrics.first_request_time else None
            ),
        }

    @classmethod
    def reset(cls):
        """Reset all metrics (useful for testing)."""
        cls._metrics = []


@contextmanager
def track_request(
    model: str,
    operation: str = "generate",
    response: Any = None,
):
    """
    Context manager to track a request with automatic timing.

    Args:
        model: Model name
        operation: Operation type (generate, chat, etc.)
        response: Optional GenerateResponse object for detailed metrics

    Example:
        >>> with track_request("qwen2.5vl:7b", "generate") as ctx:
        ...     response = client.generate("Hello!")
        ...     ctx.response = response  # For detailed metrics
    """
    start_time = time.time()
    success = False
    error = None
    ctx = type("Context", (), {"response": None})()

    try:
        yield ctx
        success = True
        response = getattr(ctx, "response", None)
    except Exception as e:
        error = str(e)
        raise
    finally:
        latency_ms = (time.time() - start_time) * 1000
        MetricsCollector.record_request(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )


def get_metrics_endpoint() -> dict[str, Any]:
    """
    Get metrics for HTTP endpoint (e.g., /metrics).

    Returns:
        JSON-serializable dictionary with current metrics
    """
    return MetricsCollector.get_metrics_json()


if __name__ == "__main__":
    # Example usage
    print("Monitoring & Metrics Example")
    print("=" * 40)

    # Simulate some requests
    with track_request("qwen2.5vl:7b", "generate"):
        time.sleep(0.1)

    with track_request("qwen2.5:14b", "chat"):
        time.sleep(0.2)

    with track_request("qwen2.5vl:7b", "generate"):
        time.sleep(0.15)

    # Get metrics
    metrics = MetricsCollector.get_metrics()
    print(f"\nTotal requests: {metrics.total_requests}")
    print(f"Successful: {metrics.successful_requests}")
    print(f"Failed: {metrics.failed_requests}")
    print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
    print(f"P95 latency: {metrics.p95_latency_ms:.2f}ms")
    print(f"\nRequests by model: {metrics.requests_by_model}")
    print(f"Requests by operation: {metrics.requests_by_operation}")

    # Get JSON metrics
    json_metrics = MetricsCollector.get_metrics_json()
    print(f"\nJSON metrics: {json_metrics}")
