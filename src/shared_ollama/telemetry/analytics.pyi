"""Type stubs for shared_ollama.telemetry.analytics module."""

from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any

class ProjectMetrics:
    project_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_by_model: dict[str, int]
    requests_by_operation: dict[str, int]
    average_latency_ms: float
    total_latency_ms: float
    last_request_time: datetime | None
    first_request_time: datetime | None

class TimeSeriesMetrics:
    timestamp: datetime
    requests_count: int
    successful_count: int
    failed_count: int
    average_latency_ms: float
    requests_by_model: dict[str, int]

class AnalyticsReport:
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_by_model: dict[str, int]
    requests_by_operation: dict[str, int]
    requests_by_project: dict[str, int]
    project_metrics: dict[str, ProjectMetrics]
    hourly_metrics: list[TimeSeriesMetrics]
    start_time: datetime | None
    end_time: datetime | None

class AnalyticsCollector:
    @classmethod
    def record_request_with_project(
        cls,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        project: str | None = ...,
        error: str | None = ...,
    ) -> None: ...
    @classmethod
    def get_analytics(
        cls,
        window_minutes: int | None = ...,
        project: str | None = ...,
    ) -> AnalyticsReport: ...
    @classmethod
    def export_json(
        cls,
        filepath: str | Path,
        window_minutes: int | None = ...,
        project: str | None = ...,
    ) -> Path: ...
    @classmethod
    def export_csv(
        cls,
        filepath: str | Path,
        window_minutes: int | None = ...,
        project: str | None = ...,
    ) -> Path: ...

def track_request_with_project(
    model: str,
    operation: str = ...,
    project: str | None = ...,
) -> AbstractContextManager[None]: ...

def get_analytics_json(
    window_minutes: int | None = ...,
    project: str | None = ...,
) -> dict[str, Any]: ...
