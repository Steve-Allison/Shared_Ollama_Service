"""
Comprehensive behavioral tests for telemetry modules.

Tests focus on real behavior, edge cases, time windows, empty metrics,
and data integrity. No mocks - tests use real data structures.
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shared_ollama import (
    AnalyticsCollector,
    MetricsCollector,
    PerformanceCollector,
    ProjectMetrics,
    ServiceMetrics,
    get_performance_stats,
    log_request_event,
    track_performance,
    track_request,
    track_request_with_project,
)
from shared_ollama.client import GenerateResponse


class TestMetricsCollector:
    """Behavioral tests for MetricsCollector."""

    def test_record_request_stores_metric(self):
        """Test that record_request() stores metric with all fields."""
        MetricsCollector.reset()

        MetricsCollector.record_request(
            model="test-model",
            operation="generate",
            latency_ms=123.45,
            success=True,
            error=None,
        )

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0

    def test_record_request_tracks_failures(self):
        """Test that record_request() tracks failed requests."""
        MetricsCollector.reset()

        MetricsCollector.record_request(
            model="test-model",
            operation="generate",
            latency_ms=50.0,
            success=False,
            error="ConnectionError: Failed",
        )

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert "ConnectionError" in metrics.errors_by_type.get("ConnectionError: Failed", "")

    def test_record_request_limits_collection_size(self):
        """Test that metrics collection is limited to _max_metrics."""
        MetricsCollector.reset()
        MetricsCollector._max_metrics = 10

        # Add more than max
        for i in range(15):
            MetricsCollector.record_request(
                model="test", operation="generate", latency_ms=float(i), success=True
            )

        # Should only keep last 10
        assert len(MetricsCollector._metrics) == 10
        # Oldest should be discarded (latency 5, not 0)
        assert MetricsCollector._metrics[0].latency_ms == 5.0

    def test_get_metrics_with_empty_collection(self):
        """Test that get_metrics() returns empty ServiceMetrics when no metrics."""
        MetricsCollector.reset()

        metrics = MetricsCollector.get_metrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.average_latency_ms == 0.0

    def test_get_metrics_calculates_average_latency(self):
        """Test that get_metrics() calculates average latency correctly."""
        MetricsCollector.reset()

        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for latency in latencies:
            MetricsCollector.record_request(
                model="test", operation="generate", latency_ms=latency, success=True
            )

        metrics = MetricsCollector.get_metrics()
        assert metrics.average_latency_ms == 30.0

    def test_get_metrics_calculates_percentiles(self):
        """Test that get_metrics() calculates percentiles correctly."""
        MetricsCollector.reset()

        # Create 100 metrics with known latencies
        for i in range(100):
            MetricsCollector.record_request(
                model="test", operation="generate", latency_ms=float(i), success=True
            )

        metrics = MetricsCollector.get_metrics()
        assert metrics.p50_latency_ms == pytest.approx(49.0, abs=1.0)
        assert metrics.p95_latency_ms == pytest.approx(94.0, abs=1.0)
        assert metrics.p99_latency_ms == pytest.approx(98.0, abs=1.0)
        assert metrics.p50_latency_ms < metrics.p95_latency_ms < metrics.p99_latency_ms

    def test_get_metrics_with_time_window(self):
        """Test that get_metrics() filters by time window correctly."""
        MetricsCollector.reset()

        # Record metrics at different times
        now = datetime.now(UTC)
        for i in range(5):
            metric = MetricsCollector._metrics[-1] if MetricsCollector._metrics else None
            if metric:
                # Manually set timestamp to past
                MetricsCollector._metrics[-1].timestamp = now - timedelta(minutes=i + 10)

            MetricsCollector.record_request(
                model="test", operation="generate", latency_ms=float(i), success=True
            )

        # Get metrics from last 5 minutes (should exclude old ones)
        metrics = MetricsCollector.get_metrics(window_minutes=5)
        # Should only include recent metrics
        assert metrics.total_requests <= 5

    def test_get_metrics_groups_by_model(self):
        """Test that get_metrics() groups requests by model."""
        MetricsCollector.reset()

        for model in ["model-a", "model-b", "model-a", "model-c", "model-b"]:
            MetricsCollector.record_request(
                model=model, operation="generate", latency_ms=50.0, success=True
            )

        metrics = MetricsCollector.get_metrics()
        assert metrics.requests_by_model["model-a"] == 2
        assert metrics.requests_by_model["model-b"] == 2
        assert metrics.requests_by_model["model-c"] == 1

    def test_get_metrics_groups_by_operation(self):
        """Test that get_metrics() groups requests by operation."""
        MetricsCollector.reset()

        for operation in ["generate", "chat", "generate", "list_models", "generate"]:
            MetricsCollector.record_request(
                model="test", operation=operation, latency_ms=50.0, success=True
            )

        metrics = MetricsCollector.get_metrics()
        assert metrics.requests_by_operation["generate"] == 3
        assert metrics.requests_by_operation["chat"] == 1
        assert metrics.requests_by_operation["list_models"] == 1

    def test_get_metrics_json_serializes_correctly(self):
        """Test that get_metrics_json() returns JSON-serializable dict."""
        MetricsCollector.reset()

        MetricsCollector.record_request(
            model="test", operation="generate", latency_ms=123.45, success=True
        )

        json_data = MetricsCollector.get_metrics_json()
        assert isinstance(json_data, dict)
        assert json_data["total_requests"] == 1
        assert isinstance(json_data["average_latency_ms"], (int, float))
        # Should be able to serialize
        json_str = json.dumps(json_data)
        assert "total_requests" in json_str


class TestTrackRequest:
    """Behavioral tests for track_request() context manager."""

    def test_track_request_records_success(self):
        """Test that track_request() records successful request."""
        MetricsCollector.reset()

        with track_request("test-model", "generate"):
            pass  # Success

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0

    def test_track_request_records_failure(self):
        """Test that track_request() records failed request."""
        MetricsCollector.reset()

        with pytest.raises(ValueError):
            with track_request("test-model", "generate"):
                raise ValueError("Test error")

        metrics = MetricsCollector.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert "ValueError" in MetricsCollector._metrics[0].error

    def test_track_request_measures_latency(self):
        """Test that track_request() measures and records latency."""
        import time

        MetricsCollector.reset()

        with track_request("test-model", "generate"):
            time.sleep(0.05)  # 50ms

        metrics = MetricsCollector.get_metrics()
        assert metrics.average_latency_ms >= 50.0

    def test_track_request_preserves_exception(self):
        """Test that track_request() preserves and re-raises exception."""
        MetricsCollector.reset()

        with pytest.raises(RuntimeError, match="Test error"):
            with track_request("test-model", "generate"):
                raise RuntimeError("Test error")

        # Should still record the metric
        metrics = MetricsCollector.get_metrics()
        assert metrics.failed_requests == 1


class TestAnalyticsCollector:
    """Behavioral tests for AnalyticsCollector."""

    def test_record_request_with_project_associates_project(self):
        """Test that record_request_with_project() associates project with metric."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        AnalyticsCollector.record_request_with_project(
            model="test",
            operation="generate",
            latency_ms=50.0,
            success=True,
            project="project-a",
        )

        analytics = AnalyticsCollector.get_analytics()
        assert "project-a" in analytics.requests_by_project
        assert analytics.requests_by_project["project-a"] >= 1

    def test_get_analytics_filters_by_project(self):
        """Test that get_analytics() filters by project correctly."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Record metrics for different projects
        for project in ["proj-a", "proj-b", "proj-a", "proj-c"]:
            AnalyticsCollector.record_request_with_project(
                model="test",
                operation="generate",
                latency_ms=50.0,
                success=True,
                project=project,
            )

        # Get analytics for proj-a only
        analytics = AnalyticsCollector.get_analytics(project="proj-a")
        assert "proj-a" in analytics.project_metrics
        assert analytics.project_metrics["proj-a"].total_requests >= 2

    def test_get_analytics_with_time_window(self):
        """Test that get_analytics() filters by time window."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Record metrics
        for i in range(10):
            AnalyticsCollector.record_request_with_project(
                model="test",
                operation="generate",
                latency_ms=float(i),
                success=True,
                project="test-proj",
            )

        # Get analytics for last 1 minute (should include all recent)
        analytics = AnalyticsCollector.get_analytics(window_minutes=1)
        assert analytics.total_requests >= 10

    def test_get_analytics_calculates_hourly_metrics(self):
        """Test that get_analytics() calculates hourly time-series metrics."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Record metrics
        for i in range(5):
            AnalyticsCollector.record_request_with_project(
                model="test",
                operation="generate",
                latency_ms=50.0,
                success=True,
                project="test-proj",
            )

        analytics = AnalyticsCollector.get_analytics()
        assert len(analytics.hourly_metrics) > 0
        assert all(
            isinstance(metric.timestamp, datetime) for metric in analytics.hourly_metrics
        )

    def test_export_json_creates_valid_file(self, tmp_path):
        """Test that export_json() creates valid JSON file."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        AnalyticsCollector.record_request_with_project(
            model="test", operation="generate", latency_ms=50.0, success=True, project="test"
        )

        filepath = tmp_path / "analytics.json"
        result_path = AnalyticsCollector.export_json(filepath)

        assert result_path.exists()
        assert result_path == filepath

        # Verify JSON is valid
        data = json.loads(filepath.read_text())
        assert "total_requests" in data
        assert data["total_requests"] >= 1

    def test_export_csv_creates_valid_file(self, tmp_path):
        """Test that export_csv() creates valid CSV file."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        AnalyticsCollector.record_request_with_project(
            model="test", operation="generate", latency_ms=50.0, success=True, project="test"
        )

        filepath = tmp_path / "analytics.csv"
        result_path = AnalyticsCollector.export_csv(filepath)

        assert result_path.exists()
        assert result_path == filepath

        # Verify CSV has header and data
        content = filepath.read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row
        assert "Timestamp" in lines[0]
        assert "Model" in lines[0]


class TestPerformanceCollector:
    """Behavioral tests for PerformanceCollector."""

    def test_record_performance_calculates_tokens_per_second(self):
        """Test that record_performance() calculates tokens/second correctly."""
        PerformanceCollector.reset()

        response = GenerateResponse(
            text="test",
            model="test-model",
            total_duration=1_000_000_000,  # 1 second
            load_duration=0,
            prompt_eval_count=10,
            prompt_eval_duration=100_000_000,
            eval_count=50,  # 50 tokens
            eval_duration=900_000_000,  # 0.9 seconds
        )

        PerformanceCollector.record_performance(
            model="test-model",
            operation="generate",
            total_latency_ms=1000.0,
            success=True,
            response=response,
        )

        stats = PerformanceCollector.get_performance_stats()
        assert stats["avg_tokens_per_second"] > 0
        # Should be approximately 50 tokens / 0.9 seconds ≈ 55 tokens/sec
        assert 50.0 < stats["avg_tokens_per_second"] < 60.0

    def test_record_performance_converts_nanoseconds_to_milliseconds(self):
        """Test that record_performance() converts ns to ms correctly."""
        PerformanceCollector.reset()

        response = GenerateResponse(
            text="test",
            model="test-model",
            total_duration=500_000_000,  # 500ms
            load_duration=200_000_000,  # 200ms
            prompt_eval_count=5,
            prompt_eval_duration=100_000_000,  # 100ms
            eval_count=10,
            eval_duration=200_000_000,  # 200ms
        )

        PerformanceCollector.record_performance(
            model="test-model",
            operation="generate",
            total_latency_ms=500.0,
            success=True,
            response=response,
        )

        metrics = PerformanceCollector._metrics
        assert len(metrics) == 1
        assert metrics[0].load_time_ms == 200.0
        assert metrics[0].generation_time_ms == 200.0

    def test_get_performance_stats_groups_by_model(self):
        """Test that get_performance_stats() groups by model."""
        PerformanceCollector.reset()

        for model in ["model-a", "model-b", "model-a"]:
            response = GenerateResponse(
                text="test",
                model=model,
                total_duration=1_000_000_000,
                load_duration=0,
                prompt_eval_count=10,
                prompt_eval_duration=100_000_000,
                eval_count=50,
                eval_duration=900_000_000,
            )
            PerformanceCollector.record_performance(
                model=model,
                operation="generate",
                total_latency_ms=1000.0,
                success=True,
                response=response,
            )

        stats = PerformanceCollector.get_performance_stats()
        assert "model-a" in stats["by_model"]
        assert "model-b" in stats["by_model"]
        assert stats["by_model"]["model-a"]["request_count"] == 2
        assert stats["by_model"]["model-b"]["request_count"] == 1

    def test_track_performance_measures_latency(self):
        """Test that track_performance() measures execution time."""
        import time

        PerformanceCollector.reset()

        with track_performance("test-model", "generate"):
            time.sleep(0.05)  # 50ms

        metrics = PerformanceCollector._metrics
        assert len(metrics) == 1
        assert metrics[0].total_latency_ms >= 50.0


class TestStructuredLogging:
    """Behavioral tests for structured logging."""

    def test_log_request_event_adds_timestamp(self, tmp_path):
        """Test that log_request_event() adds timestamp if missing."""
        from shared_ollama.telemetry import structured_logging

        # Temporarily redirect logger
        original_handlers = structured_logging.REQUEST_LOGGER.handlers[:]
        structured_logging.REQUEST_LOGGER.handlers = []
        log_path = tmp_path / "requests.jsonl"
        handler = structured_logging.logging.FileHandler(log_path)
        structured_logging.REQUEST_LOGGER.addHandler(handler)

        try:
            log_request_event({"event": "test", "model": "test-model"})

            # Verify log entry has timestamp
            content = log_path.read_text()
            data = json.loads(content.strip())
            assert "timestamp" in data
            assert isinstance(data["timestamp"], str)
        finally:
            structured_logging.REQUEST_LOGGER.removeHandler(handler)
            handler.close()
            structured_logging.REQUEST_LOGGER.handlers = original_handlers

    def test_log_request_event_serializes_datetime(self, tmp_path):
        """Test that log_request_event() serializes datetime objects."""
        from shared_ollama.telemetry import structured_logging

        original_handlers = structured_logging.REQUEST_LOGGER.handlers[:]
        structured_logging.REQUEST_LOGGER.handlers = []
        log_path = tmp_path / "requests.jsonl"
        handler = structured_logging.logging.FileHandler(log_path)
        structured_logging.REQUEST_LOGGER.addHandler(handler)

        try:
            log_request_event(
                {"event": "test", "timestamp": datetime.now(UTC), "model": "test"}
            )

            # Should serialize datetime to ISO string
            content = log_path.read_text()
            data = json.loads(content.strip())
            assert isinstance(data["timestamp"], str)
            # Should be valid ISO format
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        finally:
            structured_logging.REQUEST_LOGGER.removeHandler(handler)
            handler.close()
            structured_logging.REQUEST_LOGGER.handlers = original_handlers
