"""Type stubs for monitoring module."""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

class RequestMetrics:
    """Metrics for a single request."""
    model: str
    operation: str
    latency_ms: float
    success: bool
    error: str | None
    timestamp: datetime

class ServiceMetrics:
    """Aggregated service metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_by_model: dict[str, int]
    requests_by_operation: dict[str, int]
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    errors_by_type: dict[str, int]
    last_request_time: datetime | None
    first_request_time: datetime | None

class MetricsCollector:
    """Collects and aggregates metrics for the Ollama service."""
    
    @classmethod
    def record_request(
        cls,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        error: str | None = ...,
    ) -> None: ...
    
    @classmethod
    def get_metrics(
        cls,
        window_minutes: int | None = ...,
    ) -> ServiceMetrics: ...
    
    @classmethod
    def get_metrics_json(
        cls,
        window_minutes: int | None = ...,
    ) -> dict[str, Any]: ...
    
    @classmethod
    def reset(cls) -> None: ...

def track_request(
    model: str,
    operation: str = ...,
) -> AbstractContextManager[None]: ...

def get_metrics_endpoint() -> dict[str, Any]: ...

