"""
Enhanced Usage Analytics for Shared Ollama Service
===================================================

Advanced analytics and reporting for the central model infrastructure service.
Provides project-level tracking, time-series analysis, and export capabilities.

Usage:
    from analytics import AnalyticsCollector, track_request_with_project

    # Track request with project identifier (works with any model: qwen2.5vl:7b, qwen2.5:7b, qwen2.5:14b, granite4:tiny-h)
    with track_request_with_project("qwen2.5vl:7b", "generate", project="knowledge_machine"):
        response = client.generate("Hello!")

    # Get analytics
    analytics = AnalyticsCollector.get_analytics()
    print(f"Requests by project: {analytics.requests_by_project}")
"""

import csv
import json
import logging
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from monitoring import MetricsCollector, RequestMetrics

logger = logging.getLogger(__name__)


def _convert_datetime_to_iso(obj: Any) -> Any:
    """
    Recursively convert datetime objects to ISO format strings.

    Args:
        obj: Object to convert (can be datetime, dict, list, or primitive)

    Returns:
        Converted object with datetimes as ISO strings
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _convert_datetime_to_iso(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_datetime_to_iso(item) for item in obj]
    return obj


@dataclass
class ProjectMetrics:
    """Metrics aggregated by project."""

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


@dataclass
class TimeSeriesMetrics:
    """Time-series aggregated metrics."""

    timestamp: datetime
    requests_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    average_latency_ms: float = 0.0
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report."""

    # Overall metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0

    # Latency metrics
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Aggregations
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_operation: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_project: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Project-level metrics
    project_metrics: dict[str, ProjectMetrics] = field(default_factory=dict)

    # Time-series data
    hourly_metrics: list[TimeSeriesMetrics] = field(default_factory=list)

    # Time range
    start_time: datetime | None = None
    end_time: datetime | None = None


class AnalyticsCollector:
    """
    Enhanced analytics collector with project tracking and time-series analysis.

    Extends MetricsCollector with additional features:
    - Project-level tracking
    - Time-series aggregation
    - Export capabilities (JSON, CSV)
    - Advanced reporting
    """

    _project_metadata: ClassVar[dict[int, str]] = {}  # Maps metric index to project

    @classmethod
    def _build_metric_index_map(cls) -> dict[int, int]:
        """Build mapping from metric object id to metric index for O(1) lookup."""
        all_metrics_list = list(getattr(MetricsCollector, "_metrics", []))
        return {id(m): i for i, m in enumerate(all_metrics_list)}

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
        """
        Record a request metric with project identifier.

        Args:
            model: Model name used
            operation: Operation type (generate, chat, etc.)
            latency_ms: Request latency in milliseconds
            success: Whether request was successful
            project: Project identifier (optional)
            error: Error message if request failed
        """
        # Record in base metrics collector
        MetricsCollector.record_request(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

        # Store project metadata if provided
        if project:
            # Get the last metric index (just added)
            metrics = getattr(MetricsCollector, "_metrics", [])
            if metrics:
                # Store project association using metric index
                metric_index = len(metrics) - 1
                cls._project_metadata[metric_index] = project

    @classmethod
    def get_analytics(
        cls,
        window_minutes: int | None = None,
        project: str | None = None,
    ) -> AnalyticsReport:
        """
        Get comprehensive analytics report.

        Args:
            window_minutes: Only include metrics from last N minutes (None = all)
            project: Filter by project (None = all projects)

        Returns:
            AnalyticsReport with comprehensive statistics
        """
        # Get base metrics
        base_metrics = MetricsCollector.get_metrics(window_minutes)

        all_metrics = getattr(MetricsCollector, "_metrics", [])
        if not all_metrics:
            return AnalyticsReport()

        # Filter metrics by time window and project
        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            metrics = [m for m in all_metrics if m.timestamp >= cutoff]
        else:
            metrics = all_metrics

        # Filter by project if specified
        if project:
            metric_to_index = cls._build_metric_index_map()
            metrics = [
                m
                for m in metrics
                if cls._project_metadata.get(metric_to_index.get(id(m), -1)) == project
            ]

        if not metrics:
            return AnalyticsReport()

        # Calculate project-level metrics
        project_metrics_dict: dict[str, ProjectMetrics] = {}
        metric_to_index = cls._build_metric_index_map()

        for metric in metrics:
            metric_index = metric_to_index.get(id(metric), -1)
            proj = cls._project_metadata.get(metric_index, "unknown")

            if proj not in project_metrics_dict:
                project_metrics_dict[proj] = ProjectMetrics(project_name=proj)

            pm = project_metrics_dict[proj]
            pm.total_requests += 1
            if metric.success:
                pm.successful_requests += 1
            else:
                pm.failed_requests += 1

            pm.requests_by_model[metric.model] += 1
            pm.requests_by_operation[metric.operation] += 1
            pm.total_latency_ms += metric.latency_ms

            if not pm.first_request_time or metric.timestamp < pm.first_request_time:
                pm.first_request_time = metric.timestamp
            if not pm.last_request_time or metric.timestamp > pm.last_request_time:
                pm.last_request_time = metric.timestamp

        # Calculate averages for each project
        for pm in project_metrics_dict.values():
            if pm.total_requests > 0:
                pm.average_latency_ms = pm.total_latency_ms / pm.total_requests

        # Calculate time-series (hourly buckets)
        hourly_metrics = cls._calculate_hourly_metrics(metrics)

        # Calculate requests by project
        requests_by_project = {
            pm.project_name: pm.total_requests for pm in project_metrics_dict.values()
        }

        # Calculate success rate
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
            start_time=min(m.timestamp for m in metrics),
            end_time=max(m.timestamp for m in metrics),
        )

    @classmethod
    def _calculate_hourly_metrics(cls, metrics: list[RequestMetrics]) -> list[TimeSeriesMetrics]:
        """Calculate hourly aggregated metrics."""
        if not metrics:
            return []

        # Group by hour
        hourly_data: dict[datetime, list[RequestMetrics]] = defaultdict(list)

        for metric in metrics:
            # Round to nearest hour
            hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour].append(metric)

        # Calculate metrics for each hour
        hourly_metrics = []
        for hour, hour_metrics in sorted(hourly_data.items()):
            total_latency = sum(m.latency_ms for m in hour_metrics)
            avg_latency = total_latency / len(hour_metrics) if hour_metrics else 0.0

            requests_by_model = defaultdict(int)
            for m in hour_metrics:
                requests_by_model[m.model] += 1

            hourly_metrics.append(
                TimeSeriesMetrics(
                    timestamp=hour,
                    requests_count=len(hour_metrics),
                    successful_count=sum(1 for m in hour_metrics if m.success),
                    failed_count=sum(1 for m in hour_metrics if not m.success),
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
        """
        Export analytics to JSON file.

        Args:
            filepath: Path to output JSON file
            window_minutes: Time window for metrics
            project: Filter by project

        Returns:
            Path to exported file
        """
        analytics = cls.get_analytics(window_minutes, project)

        # Convert to dict for JSON serialization
        data = asdict(analytics)

        # Convert datetime objects to ISO format strings
        data = _convert_datetime_to_iso(data)

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported analytics to {path}")
        return path

    @classmethod
    def export_csv(
        cls,
        filepath: str | Path,
        window_minutes: int | None = None,
        project: str | None = None,
    ) -> Path:
        """
        Export analytics to CSV file.

        Args:
            filepath: Path to output CSV file
            window_minutes: Time window for metrics
            project: Filter by project

        Returns:
            Path to exported file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "Timestamp",
                "Model",
                "Operation",
                "Latency (ms)",
                "Success",
                "Error",
                "Project",
            ])

            # Write data rows
            metrics = getattr(MetricsCollector, "_metrics", [])
            if window_minutes:
                cutoff = datetime.now() - timedelta(minutes=window_minutes)
                metrics = [m for m in metrics if m.timestamp >= cutoff]

            metric_to_index = cls._build_metric_index_map()
            for metric in metrics:
                metric_index = metric_to_index.get(id(metric), -1)
                proj = cls._project_metadata.get(metric_index, "unknown")
                if project and proj != project:
                    continue

                writer.writerow([
                    metric.timestamp.isoformat(),
                    metric.model,
                    metric.operation,
                    f"{metric.latency_ms:.2f}",
                    "Yes" if metric.success else "No",
                    metric.error or "",
                    proj,
                ])

        logger.info(f"Exported analytics to CSV: {path}")
        return path


@contextmanager
def track_request_with_project(
    model: str,
    operation: str = "generate",
    project: str | None = None,
) -> Generator[None, None, None]:
    """
    Context manager to track a request with project identifier.

    Args:
        model: Model name
        operation: Operation type (generate, chat, etc.)
        project: Project identifier (optional)

    Example:
        >>> with track_request_with_project(
        ...     "qwen2.5vl:7b", "generate", project="knowledge_machine"
        ... ):
        ...     response = client.generate("Hello!")
    """
    start_time = time.perf_counter()
    success = False
    error = None

    try:
        yield
        success = True
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
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
    """
    Get analytics as JSON-serializable dictionary.

    Args:
        window_minutes: Time window for metrics
        project: Filter by project

    Returns:
        Dictionary with analytics
    """
    analytics = AnalyticsCollector.get_analytics(window_minutes, project)

    # Convert to dict and handle datetime serialization
    data = asdict(analytics)
    return _convert_datetime_to_iso(data)  # type: ignore[return-value]
