"""Enhanced usage analytics for the Shared Ollama Service.

This module provides project-based analytics tracking and reporting with
time-series aggregation and export capabilities. Extends MetricsCollector
with project tracking and comprehensive analytics.

Key Features:
    - Project-Based Tracking: Associates requests with projects via X-Project-Name header
    - Time-Series Aggregation: Hourly metrics for trend analysis
    - Comprehensive Reports: Percentiles, success rates, model/operation breakdowns
    - Export Functionality: JSON and CSV export for external analysis
    - Efficient Filtering: Filter by time window and/or project

Design Principles:
    - Extends MetricsCollector: Builds on base metrics collection
    - Project Association: Maps metrics to projects via index tracking
    - Time-Series Analysis: Hourly aggregation for trend visualization
    - Export Formats: Multiple formats (JSON, CSV) for different use cases
    - Performance: Efficient filtering and aggregation algorithms

Analytics Features:
    - Project-Level Metrics: Aggregated statistics per project
    - Hourly Time-Series: Request counts and latencies by hour
    - Success Rates: Overall and per-project success rates
    - Latency Percentiles: p50, p95, p99 across all metrics
    - Model Breakdowns: Request counts by model
    - Operation Breakdowns: Request counts by operation type

Project Tracking:
    - Projects identified via X-Project-Name HTTP header
    - Project metadata stored separately from metrics for efficiency
    - Unknown projects tracked as "unknown"
    - Project filtering enables per-project analytics
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import TypeAdapter

from shared_ollama.telemetry.metrics import MetricsCollector

if TYPE_CHECKING:
    from collections.abc import Generator

    from shared_ollama.telemetry.metrics import RequestMetrics
else:  # pragma: no cover - used only for runtime type hint evaluation
    Generator = Any  # type: ignore[assignment]
    RequestMetrics = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _convert_datetime_to_iso(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings.

    Traverses nested data structures (dicts, lists, defaultdicts) and
    converts all datetime objects to ISO 8601 strings. Uses Pydantic's
    built-in serialization for datetime objects to ensure proper timezone
    handling.

    Args:
        obj: Object to convert. Can be datetime, dict, list, defaultdict,
            or any other type.

    Returns:
        Object with all datetime values converted to ISO strings.
        Other types are returned unchanged.

    Side effects:
        None. Pure function.
    """
    match obj:
        case datetime():
            # Use Pydantic's datetime serialization for proper timezone handling
            return TypeAdapter(datetime).dump_python(obj, mode="json")
        case defaultdict():
            return {k: _convert_datetime_to_iso(v) for k, v in obj.items()}
        case dict():
            return {k: _convert_datetime_to_iso(v) for k, v in obj.items()}
        case list():
            return [_convert_datetime_to_iso(item) for item in obj]
        case _:
            return obj


@dataclass(slots=True)
class ProjectMetrics:
    """Metrics aggregated by project.

    Contains request statistics for a single project. All time values
    are in milliseconds. Timestamps are timezone-aware (UTC).

    Attributes:
        project_name: Project identifier (from X-Project-Name header).
        total_requests: Total number of requests for this project.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        requests_by_model: Dictionary mapping model names to request counts.
        requests_by_operation: Dictionary mapping operation types to request counts.
        average_latency_ms: Average request latency in milliseconds.
        total_latency_ms: Cumulative latency for all requests (ms).
        last_request_time: Timestamp of most recent request. None if no requests.
        first_request_time: Timestamp of oldest request. None if no requests.
    """

    project_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_operation: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    average_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    last_request_time: datetime | None = None
    first_request_time: datetime | None = None


@dataclass(slots=True)
class TimeSeriesMetrics:
    """Time series metrics for a specific time period.

    Contains aggregated metrics for a single hour. Used for time-series
    analysis and trend visualization.

    Attributes:
        timestamp: Hour timestamp (minute/second/microsecond set to 0).
        requests_count: Total number of requests in this hour.
        successful_count: Number of successful requests.
        failed_count: Number of failed requests.
        average_latency_ms: Average request latency in milliseconds.
        requests_by_model: Dictionary mapping model names to request counts.
    """

    timestamp: datetime
    requests_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    average_latency_ms: float = 0.0
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass(slots=True)
class AnalyticsReport:
    """Comprehensive analytics report.

    Contains aggregated analytics across all projects, models, and operations.
    Includes time-series data and project-level breakdowns.

    Attributes:
        total_requests: Total number of requests in the report window.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        success_rate: Success rate as a float (0.0-1.0).
        average_latency_ms: Average request latency in milliseconds.
        p50_latency_ms: 50th percentile (median) latency in milliseconds.
        p95_latency_ms: 95th percentile latency in milliseconds.
        p99_latency_ms: 99th percentile latency in milliseconds.
        requests_by_model: Dictionary mapping model names to request counts.
        requests_by_operation: Dictionary mapping operation types to request counts.
        requests_by_project: Dictionary mapping project names to request counts.
        project_metrics: Dictionary mapping project names to ProjectMetrics objects.
        hourly_metrics: List of TimeSeriesMetrics, one per hour in the window.
        start_time: Timestamp of earliest request in the window. None if no requests.
        end_time: Timestamp of latest request in the window. None if no requests.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_operation: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_project: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    project_metrics: dict[str, ProjectMetrics] = field(default_factory=dict)
    hourly_metrics: list[TimeSeriesMetrics] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None


class AnalyticsCollector:
    """Collects and analyzes usage analytics with project tracking.

    Extends MetricsCollector with project-based tracking and comprehensive
    analytics reporting. Maintains a mapping from metric indices to project
    names for efficient filtering and aggregation.

    This collector implements AnalyticsCollectorInterface and is used by
    AnalyticsCollectorAdapter in the infrastructure layer.

    Attributes:
        _project_metadata: Class variable mapping metric indices to project names.
            Used to associate metrics with projects after they're recorded.
            Index corresponds to position in MetricsCollector._metrics list.

    Thread Safety:
        Not thread-safe by design. This collector is intended for use in
        single-threaded async applications. If multi-threaded access is needed,
        add locks around _project_metadata operations.

    Project Tracking:
        Projects are associated with metrics via index mapping. When a metric
        is recorded with a project, the metric's index in MetricsCollector._metrics
        is mapped to the project name in _project_metadata. This enables efficient
        filtering without modifying the base metrics structure.

    Analytics Capabilities:
        - Project-level aggregation and breakdowns
        - Hourly time-series metrics for trend analysis
        - Comprehensive reports with percentiles and success rates
        - JSON and CSV export for external analysis
        - Time window and project filtering
    """

    _project_metadata: ClassVar[dict[int, str]] = {}

    @classmethod
    def _build_metric_index_map(cls) -> dict[int, int]:
        """Build a mapping from metric ID to index.

        Creates a dictionary that maps the id() of each metric object to
        its index in the MetricsCollector._metrics list. Used for efficient
        project filtering.

        Returns:
            Dictionary mapping metric object IDs to list indices.
        """
        all_metrics_list = list(getattr(MetricsCollector, "_metrics", []))
        return {id(metric): index for index, metric in enumerate(all_metrics_list)}

    @classmethod
    def _filter_by_window(
        cls, metrics: list[RequestMetrics], window_minutes: int | None
    ) -> list[RequestMetrics]:
        if not window_minutes:
            return metrics
        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
        return [metric for metric in metrics if metric.timestamp >= cutoff]

    @classmethod
    def _filter_by_project(
        cls, metrics: list[RequestMetrics], project: str | None
    ) -> list[RequestMetrics]:
        if not project:
            return metrics
        metric_to_index = cls._build_metric_index_map()
        return [
            metric
            for metric in metrics
            if cls._project_metadata.get(metric_to_index.get(id(metric), -1)) == project
        ]

    @classmethod
    def _aggregate_project_metrics(
        cls, metrics: list[RequestMetrics]
    ) -> dict[str, ProjectMetrics]:
        metric_to_index = cls._build_metric_index_map()
        project_metrics_dict: dict[str, ProjectMetrics] = {}
        for metric in metrics:
            metric_index = metric_to_index.get(id(metric), -1)
            proj = cls._project_metadata.get(metric_index, "unknown")
            pm = project_metrics_dict.setdefault(proj, ProjectMetrics(project_name=proj))
            pm.total_requests += 1
            pm.successful_requests += int(metric.success)
            pm.failed_requests += int(not metric.success)
            pm.requests_by_model[metric.model] += 1
            pm.requests_by_operation[metric.operation] += 1
            pm.total_latency_ms += metric.latency_ms
            if not pm.first_request_time or metric.timestamp < pm.first_request_time:
                pm.first_request_time = metric.timestamp
            if not pm.last_request_time or metric.timestamp > pm.last_request_time:
                pm.last_request_time = metric.timestamp
        for pm in project_metrics_dict.values():
            if pm.total_requests > 0:
                pm.average_latency_ms = pm.total_latency_ms / pm.total_requests
        return project_metrics_dict

    @classmethod
    def record_request_with_project(
        cls,
        model: str,
        operation: str,
        latency_ms: float,
        success: bool,
        project: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a request metric with project tracking.

        Records the metric via MetricsCollector and associates it with
        a project name if provided. The project association is stored
        separately for efficient filtering.

        Args:
            model: Model name used for the request.
            operation: Operation type (e.g., "generate", "chat").
            latency_ms: Request latency in milliseconds (>=0.0).
            success: Whether the request succeeded.
            project: Project name for tracking. None if not provided.
            error: Error message if request failed. None if successful.

        Side effects:
            - Calls MetricsCollector.record_request()
            - Updates _project_metadata if project is provided
        """
        MetricsCollector.record_request(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

        if project:
            metrics = getattr(MetricsCollector, "_metrics", [])
            if metrics:
                metric_index = len(metrics) - 1
                cls._project_metadata[metric_index] = project

    @classmethod
    def get_analytics(
        cls,
        window_minutes: int | None = None,
        project: str | None = None,
    ) -> AnalyticsReport:
        """Get comprehensive analytics report.

        Computes aggregated analytics with optional filtering by time window
        and project. Includes project-level breakdowns and hourly time-series
        data.

        Args:
            window_minutes: Time window in minutes. If None, aggregates all
                metrics. Only metrics within the window are included.
            project: Filter by project name. If None, includes all projects.
                Only metrics for the specified project are included.

        Returns:
            AnalyticsReport with comprehensive statistics. Returns empty
            AnalyticsReport if no metrics match the filters.

        Side effects:
            - Filters metrics by timestamp if window_minutes specified
            - Filters metrics by project if project specified
            - Computes aggregations (O(n) for filtering, O(n log n) for sorting)
        """
        base_metrics = MetricsCollector.get_metrics(window_minutes)

        all_metrics = getattr(MetricsCollector, "_metrics", [])
        if not all_metrics:
            return AnalyticsReport()

        metrics = cls._filter_by_window(all_metrics, window_minutes)
        metrics = cls._filter_by_project(metrics, project)

        if not metrics:
            return AnalyticsReport()

        project_metrics_dict = cls._aggregate_project_metrics(metrics)
        hourly_metrics = cls._calculate_hourly_metrics(metrics)

        requests_by_project = {
            pm.project_name: pm.total_requests for pm in project_metrics_dict.values()
        }

        success_rate = (
            base_metrics.successful_requests / base_metrics.total_requests
            if base_metrics.total_requests > 0
            else 0.0
        )

        return AnalyticsReport(
            total_requests=base_metrics.total_requests,
            successful_requests=base_metrics.successful_requests,
            failed_requests=base_metrics.failed_requests,
            success_rate=success_rate,
            average_latency_ms=base_metrics.average_latency_ms,
            p50_latency_ms=base_metrics.p50_latency_ms,
            p95_latency_ms=base_metrics.p95_latency_ms,
            p99_latency_ms=base_metrics.p99_latency_ms,
            requests_by_model=base_metrics.requests_by_model,
            requests_by_operation=base_metrics.requests_by_operation,
            requests_by_project=requests_by_project,
            project_metrics=project_metrics_dict,
            hourly_metrics=hourly_metrics,
            start_time=min(metric.timestamp for metric in metrics),
            end_time=max(metric.timestamp for metric in metrics),
        )

    @classmethod
    def _calculate_hourly_metrics(cls, metrics: list[RequestMetrics]) -> list[TimeSeriesMetrics]:
        """Calculate hourly aggregated metrics.

        Groups metrics by hour and computes aggregated statistics for each
        hour. Hours are determined by rounding timestamps down to the hour.

        Args:
            metrics: List of request metrics to aggregate.

        Returns:
            List of TimeSeriesMetrics, one per hour that contains metrics.
            Sorted by timestamp (earliest first). Returns empty list if
            no metrics provided.
        """
        if not metrics:
            return []

        hourly_data: defaultdict[datetime, list[RequestMetrics]] = defaultdict(list)

        for metric in metrics:
            hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour].append(metric)

        hourly_metrics: list[TimeSeriesMetrics] = []
        for hour, hour_metrics in sorted(hourly_data.items()):
            total_latency = sum(metric.latency_ms for metric in hour_metrics)
            avg_latency = total_latency / len(hour_metrics) if hour_metrics else 0.0

            requests_by_model: defaultdict[str, int] = defaultdict(int)
            for metric in hour_metrics:
                requests_by_model[metric.model] += 1

            hourly_metrics.append(
                TimeSeriesMetrics(
                    timestamp=hour,
                    requests_count=len(hour_metrics),
                    successful_count=sum(1 for metric in hour_metrics if metric.success),
                    failed_count=sum(1 for metric in hour_metrics if not metric.success),
                    average_latency_ms=avg_latency,
                    requests_by_model=dict(requests_by_model),
                )
            )

        return hourly_metrics

    @classmethod
    def export_json(
        cls,
        filepath: str | Path,
        window_minutes: int | None = None,
        project: str | None = None,
    ) -> Path:
        """Export analytics to JSON file.

        Generates a comprehensive analytics report and writes it to a JSON
        file with pretty-printed formatting. All datetime objects are
        converted to ISO 8601 strings.

        Args:
            filepath: Path to output file. Parent directories are created
                if they don't exist.
            window_minutes: Time window in minutes. If None, exports all metrics.
            project: Filter by project name. If None, exports all projects.

        Returns:
            Path to exported file (absolute path).

        Side effects:
            - Creates parent directories if they don't exist
            - Writes JSON file to disk
            - Logs info message with file path
        """
        analytics = cls.get_analytics(window_minutes, project)

        data: dict[str, Any] = {
            "total_requests": analytics.total_requests,
            "successful_requests": analytics.successful_requests,
            "failed_requests": analytics.failed_requests,
            "success_rate": analytics.success_rate,
            "average_latency_ms": analytics.average_latency_ms,
            "p50_latency_ms": analytics.p50_latency_ms,
            "p95_latency_ms": analytics.p95_latency_ms,
            "p99_latency_ms": analytics.p99_latency_ms,
            "requests_by_model": dict(analytics.requests_by_model),
            "requests_by_operation": dict(analytics.requests_by_operation),
            "requests_by_project": dict(analytics.requests_by_project),
            "project_metrics": {
                key: {
                    "project_name": value.project_name,
                    "total_requests": value.total_requests,
                    "successful_requests": value.successful_requests,
                    "failed_requests": value.failed_requests,
                    "requests_by_model": dict(value.requests_by_model),
                    "requests_by_operation": dict(value.requests_by_operation),
                    "average_latency_ms": value.average_latency_ms,
                    "total_latency_ms": value.total_latency_ms,
                    "last_request_time": value.last_request_time,
                    "first_request_time": value.first_request_time,
                }
                for key, value in analytics.project_metrics.items()
            },
            "hourly_metrics": [
                {
                    "timestamp": hour.timestamp,
                    "requests_count": hour.requests_count,
                    "successful_count": hour.successful_count,
                    "failed_count": hour.failed_count,
                    "average_latency_ms": hour.average_latency_ms,
                    "requests_by_model": dict(hour.requests_by_model),
                }
                for hour in analytics.hourly_metrics
            ],
            "start_time": analytics.start_time,
            "end_time": analytics.end_time,
        }

        data = _convert_datetime_to_iso(data)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        logger.info("Exported analytics to %s", path)
        return path

    @classmethod
    def export_csv(
        cls,
        filepath: str | Path,
        window_minutes: int | None = None,
        project: str | None = None,
    ) -> Path:
        """Export analytics to CSV file.

        Exports raw metrics data to CSV format with one row per request.
        Includes project information for each request.

        Args:
            filepath: Path to output file. Parent directories are created
                if they don't exist.
            window_minutes: Time window in minutes. If None, exports all metrics.
            project: Filter by project name. If None, exports all projects.

        Returns:
            Path to exported file (absolute path).

        Side effects:
            - Creates parent directories if they don't exist
            - Writes CSV file to disk
            - Logs info message with file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Timestamp",
                    "Model",
                    "Operation",
                    "Latency (ms)",
                    "Success",
                    "Error",
                    "Project",
                ]
            )

            metrics = getattr(MetricsCollector, "_metrics", [])
            if window_minutes:
                cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
                metrics = [metric for metric in metrics if metric.timestamp >= cutoff]

            metric_to_index = cls._build_metric_index_map()
            for metric in metrics:
                metric_index = metric_to_index.get(id(metric), -1)
                proj = cls._project_metadata.get(metric_index, "unknown")
                if project and proj != project:
                    continue

                writer.writerow(
                    [
                        metric.timestamp.isoformat(),
                        metric.model,
                        metric.operation,
                        f"{metric.latency_ms:.2f}",
                        "Yes" if metric.success else "No",
                        metric.error or "",
                        proj,
                    ]
                )

        logger.info("Exported analytics to CSV: %s", path)
        return path


@contextmanager
def track_request_with_project(
    model: str,
    operation: str = "generate",
    project: str | None = None,
) -> Generator[None]:
    """Context manager to track a request with project tracking.

    Automatically measures execution time and records metrics with project
    association. Handles exceptions by recording error information.

    Args:
        model: Model name for the request.
        operation: Operation type (e.g., "generate", "chat"). Defaults to "generate".
        project: Project name for tracking. None if not provided.

    Yields:
        None. The context manager tracks timing while the context is active.

    Side effects:
        - Measures execution time using time.perf_counter()
        - Records metrics via AnalyticsCollector.record_request_with_project()
        - Logs error information if exception occurs

    Example:
        >>> with track_request_with_project("qwen3-vl:8b-instruct-q4_K_M", "generate", "my-project"):
        ...     result = await client.generate("Hello")
    """
    start_time = time.perf_counter()
    success = False
    error = None

    try:
        yield
        success = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        AnalyticsCollector.record_request_with_project(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            project=project,
            error=error,
        )


def get_analytics_json(
    window_minutes: int | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Get analytics as JSON-serializable dictionary.

    Convenience function that generates an analytics report and converts
    it to a dictionary with all datetime objects converted to ISO strings.

    Args:
        window_minutes: Time window in minutes. If None, returns all metrics.
        project: Filter by project name. If None, returns all projects.

    Returns:
        Dictionary with analytics data. All datetime values are ISO 8601
        strings. Structure matches AnalyticsReport dataclass fields.
    """
    analytics = AnalyticsCollector.get_analytics(window_minutes, project)
    data = {
        "total_requests": analytics.total_requests,
        "successful_requests": analytics.successful_requests,
        "failed_requests": analytics.failed_requests,
        "success_rate": analytics.success_rate,
        "average_latency_ms": analytics.average_latency_ms,
        "p50_latency_ms": analytics.p50_latency_ms,
        "p95_latency_ms": analytics.p95_latency_ms,
        "p99_latency_ms": analytics.p99_latency_ms,
        "requests_by_model": dict(analytics.requests_by_model),
        "requests_by_operation": dict(analytics.requests_by_operation),
        "requests_by_project": dict(analytics.requests_by_project),
        "project_metrics": {
            k: {
                "project_name": v.project_name,
                "total_requests": v.total_requests,
                "successful_requests": v.successful_requests,
                "failed_requests": v.failed_requests,
                "requests_by_model": dict(v.requests_by_model),
                "requests_by_operation": dict(v.requests_by_operation),
                "average_latency_ms": v.average_latency_ms,
                "total_latency_ms": v.total_latency_ms,
                "last_request_time": v.last_request_time,
                "first_request_time": v.first_request_time,
            }
            for k, v in analytics.project_metrics.items()
        },
        "hourly_metrics": [
            {
                "timestamp": h.timestamp,
                "requests_count": h.requests_count,
                "successful_count": h.successful_count,
                "failed_count": h.failed_count,
                "average_latency_ms": h.average_latency_ms,
                "requests_by_model": dict(h.requests_by_model),
            }
            for h in analytics.hourly_metrics
        ],
        "start_time": analytics.start_time,
        "end_time": analytics.end_time,
    }
    return _convert_datetime_to_iso(data)  # type: ignore[return-value]


__all__ = [
    "AnalyticsCollector",
    "AnalyticsReport",
    "ProjectMetrics",
    "TimeSeriesMetrics",
    "get_analytics_json",
    "track_request_with_project",
]
