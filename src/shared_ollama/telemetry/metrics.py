"""
Metrics collection and monitoring utilities for the Shared Ollama Service.
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


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
    """Collects and aggregates metrics for the Ollama service."""

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
        if not cls._metrics:
            return ServiceMetrics()

        if window_minutes:
            cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
            metrics = [m for m in cls._metrics if m.timestamp >= cutoff]
        else:
            metrics = cls._metrics

        if not metrics:
            return ServiceMetrics()

        latencies = [m.latency_ms for m in metrics]
        latencies_sorted = sorted(latencies)

        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        failed = total - successful

        requests_by_model = defaultdict(int)
        requests_by_operation = defaultdict(int)
        errors_by_type = defaultdict(int)

        for metric in metrics:
            requests_by_model[metric.model] += 1
            requests_by_operation[metric.operation] += 1
            if metric.error:
                errors_by_type[metric.error] += 1

        if len(latencies_sorted) >= 2:
            quantiles = statistics.quantiles(latencies_sorted, n=100)
            p50 = quantiles[49]
            p95 = quantiles[94]
            p99 = quantiles[98]
        elif len(latencies_sorted) == 1:
            p50 = p95 = p99 = latencies_sorted[0]
        else:
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
    def reset(cls) -> None:
        cls._metrics = []


@contextmanager
def track_request(model: str, operation: str) -> Generator[None, None, None]:
    """
    Context manager to track a request's latency.
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
    """
    Convenience helper for exposing metrics via an HTTP endpoint.
    """

    return MetricsCollector.get_metrics_json()


__all__ = ["MetricsCollector", "ServiceMetrics", "get_metrics_endpoint", "track_request"]

