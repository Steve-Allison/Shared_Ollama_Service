"""
Enhanced usage analytics for the Shared Ollama Service.
"""

from __future__ import annotations

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

from shared_ollama.telemetry.metrics import MetricsCollector, RequestMetrics

logger = logging.getLogger(__name__)


def _convert_datetime_to_iso(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, defaultdict):
        return {k: _convert_datetime_to_iso(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _convert_datetime_to_iso(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_datetime_to_iso(item) for item in obj]
    return obj


@dataclass
class ProjectMetrics:
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
    timestamp: datetime
    requests_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    average_latency_ms: float = 0.0
    requests_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class AnalyticsReport:
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
    _project_metadata: ClassVar[dict[int, str]] = {}

    @classmethod
    def _build_metric_index_map(cls) -> dict[int, int]:
        all_metrics_list = list(getattr(MetricsCollector, "_metrics", []))
        return {id(metric): index for index, metric in enumerate(all_metrics_list)}

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
        base_metrics = MetricsCollector.get_metrics(window_minutes)

        all_metrics = getattr(MetricsCollector, "_metrics", [])
        if not all_metrics:
            return AnalyticsReport()

        if window_minutes:
            cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
            metrics = [metric for metric in all_metrics if metric.timestamp >= cutoff]
        else:
            metrics = all_metrics

        if project:
            metric_to_index = cls._build_metric_index_map()
            metrics = [
                metric
                for metric in metrics
                if cls._project_metadata.get(metric_to_index.get(id(metric), -1)) == project
            ]

        if not metrics:
            return AnalyticsReport()

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

        for pm in project_metrics_dict.values():
            if pm.total_requests > 0:
                pm.average_latency_ms = pm.total_latency_ms / pm.total_requests

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
        if not metrics:
            return []

        hourly_data: dict[datetime, list[RequestMetrics]] = defaultdict(list)

        for metric in metrics:
            hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour].append(metric)

        hourly_metrics: list[TimeSeriesMetrics] = []
        for hour, hour_metrics in sorted(hourly_data.items()):
            total_latency = sum(metric.latency_ms for metric in hour_metrics)
            avg_latency = total_latency / len(hour_metrics) if hour_metrics else 0.0

            requests_by_model = defaultdict(int)
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
        analytics = cls.get_analytics(window_minutes, project)

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
) -> Generator[None, None, None]:
    start_time = time.perf_counter()
    success = False
    error = None

    try:
        yield
        success = True
    except Exception as exc:  # noqa: BLE001
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
    analytics = AnalyticsCollector.get_analytics(window_minutes, project)
    data = asdict(analytics)
    return _convert_datetime_to_iso(data)  # type: ignore[return-value]


__all__ = [
    "AnalyticsCollector",
    "AnalyticsReport",
    "ProjectMetrics",
    "TimeSeriesMetrics",
    "get_analytics_json",
    "track_request_with_project",
]

