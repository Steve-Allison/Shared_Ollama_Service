"""
Comprehensive tests for Python 3.13+ modernization improvements.

Tests cover:
- Native feature implementations (statistics.quantiles, UTC timestamps, perf_counter)
- Enhanced error handling (JSON, HTTP, File I/O)
- Edge cases and validation
- Analytics and performance logging
- Resilience features
"""

import json
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from shared_ollama import (
    AnalyticsCollector,
    CircuitBreaker,
    CircuitState,
    GenerateResponse,
    MetricsCollector,
    PerformanceCollector,
    SharedOllamaClient,
    exponential_backoff_retry,
    track_performance,
    track_request,
    track_request_with_project,
)


class TestNativeFeatures:
    """Tests for Python 3.13+ native feature implementations."""

    def test_statistics_quantiles_with_multiple_datapoints(self):
        """Test statistics.quantiles() with sufficient data points."""
        # Record multiple metrics
        MetricsCollector.reset()
        for i in range(100):
            MetricsCollector.record_request(
                model="qwen2.5vl:7b",
                operation="generate",
                latency_ms=float(i),
                success=True,
            )

        metrics = MetricsCollector.get_metrics()

        # Verify percentiles are calculated correctly
        assert metrics.p50_latency_ms > 0
        assert metrics.p95_latency_ms > 0
        assert metrics.p99_latency_ms > 0
        assert metrics.p50_latency_ms < metrics.p95_latency_ms < metrics.p99_latency_ms

    def test_statistics_quantiles_single_datapoint_edge_case(self):
        """Test edge case: single data point for quantiles."""
        MetricsCollector.reset()
        MetricsCollector.record_request(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=100.0,
            success=True,
        )

        metrics = MetricsCollector.get_metrics()

        # All percentiles should equal the single value
        assert metrics.p50_latency_ms == 100.0
        assert metrics.p95_latency_ms == 100.0
        assert metrics.p99_latency_ms == 100.0

    def test_statistics_quantiles_empty_metrics(self):
        """Test edge case: no metrics."""
        MetricsCollector.reset()
        metrics = MetricsCollector.get_metrics()

        # Should return default values
        assert metrics.total_requests == 0
        assert metrics.p50_latency_ms == 0.0
        assert metrics.p95_latency_ms == 0.0
        assert metrics.p99_latency_ms == 0.0

    def test_utc_timestamps(self):
        """Test that timestamps use UTC timezone."""
        MetricsCollector.reset()
        MetricsCollector.record_request(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=50.0,
            success=True,
        )

        metrics = getattr(MetricsCollector, "_metrics", [])
        assert len(metrics) > 0

        # Verify timestamp is timezone-aware and in UTC
        timestamp = metrics[0].timestamp
        assert timestamp.tzinfo is not None
        assert timestamp.tzinfo == UTC

    def test_perf_counter_timing(self):
        """Test that perf_counter is used for accurate timing."""
        MetricsCollector.reset()

        # Use track_request context manager which uses perf_counter
        with track_request("qwen2.5vl:7b", "generate"):
            time.sleep(0.01)  # Sleep for 10ms

        metrics = MetricsCollector.get_metrics()
        # Latency should be at least 10ms (accounting for some overhead)
        assert metrics.average_latency_ms >= 10.0


class TestErrorHandling:
    """Tests for enhanced error handling."""

    def test_json_decode_error_in_list_models(self, mock_client):
        """Test JSON decode error handling in list_models()."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_client.session.get.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            mock_client.list_models()

    def test_json_decode_error_in_generate(self, mock_client):
        """Test JSON decode error handling in generate()."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_client.session.post.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            mock_client.generate("Hello!")

    def test_http_error_in_list_models(self, mock_client):
        """Test HTTP error handling in list_models()."""
        mock_response = Mock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError("500 Server Error")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_client.session.get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            mock_client.list_models()

    def test_invalid_response_structure_in_list_models(self, mock_client):
        """Test validation of response structure in list_models()."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = ["not", "a", "dict"]  # Invalid structure
        mock_client.session.get.return_value = mock_response

        with pytest.raises(ValueError, match="Expected dict response"):
            mock_client.list_models()

    def test_health_check_logs_debug_on_failure(self, mock_client):
        """Test that health_check logs debug message on failure."""
        mock_client.session.get.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )

        with patch("shared_ollama.client.sync.logger") as mock_logger:
            result = mock_client.health_check()
            assert result is False
            mock_logger.debug.assert_called_once()

    @pytest.mark.skip(reason="Permission errors are environment-dependent")
    def test_file_io_permission_error_export_json(self):
        """Test file I/O permission error handling in export_json()."""
        AnalyticsCollector._project_metadata.clear()
        MetricsCollector.reset()

        # Try to write to a protected location
        with pytest.raises((PermissionError, OSError)):
            AnalyticsCollector.export_json("/root/protected/analytics.json")

    @pytest.mark.skip(reason="Permission errors are environment-dependent")
    def test_file_io_permission_error_export_csv(self):
        """Test file I/O permission error handling in export_csv()."""
        AnalyticsCollector._project_metadata.clear()
        MetricsCollector.reset()

        # Try to write to a protected location
        with pytest.raises((PermissionError, OSError)):
            AnalyticsCollector.export_csv("/root/protected/analytics.csv")


class TestAnalyticsExport:
    """Tests for analytics export functionality."""

    def test_export_json_creates_file(self):
        """Test that export_json creates a valid JSON file."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Record some test metrics
        for i in range(10):
            AnalyticsCollector.record_request_with_project(
                model="qwen2.5vl:7b",
                operation="generate",
                latency_ms=float(i * 10),
                success=True,
                project="test_project",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "analytics.json"
            result_path = AnalyticsCollector.export_json(filepath)

            assert result_path.exists()
            assert result_path == filepath

            # Verify JSON content
            with filepath.open("r") as f:
                data = json.load(f)
                # Total requests includes all metrics (base_metrics calculation)
                assert data["total_requests"] >= 10
                assert "test_project" in data["requests_by_project"]
                assert data["requests_by_project"]["test_project"] >= 10

    def test_export_csv_creates_file(self):
        """Test that export_csv creates a valid CSV file."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Record some test metrics
        for i in range(5):
            AnalyticsCollector.record_request_with_project(
                model="qwen2.5vl:7b",
                operation="generate",
                latency_ms=float(i * 10),
                success=True,
                project="csv_project",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "analytics.csv"
            result_path = AnalyticsCollector.export_csv(filepath)

            assert result_path.exists()
            assert result_path == filepath

            # Verify CSV content
            with filepath.open("r") as f:
                lines = f.readlines()
                assert len(lines) > 1  # Header + data rows
                assert "Timestamp" in lines[0]
                assert "csv_project" in "".join(lines)

    def test_export_json_datetime_serialization(self):
        """Test that datetime objects are properly serialized in JSON export."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Need at least 2 requests for analytics to have start/end times
        AnalyticsCollector.record_request_with_project(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=50.0,
            success=True,
            project="datetime_test",
        )
        AnalyticsCollector.record_request_with_project(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=60.0,
            success=True,
            project="datetime_test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "analytics.json"
            AnalyticsCollector.export_json(filepath)

            # Verify datetime fields are ISO strings
            with filepath.open("r") as f:
                data = json.load(f)
                # Check that timestamps are strings (ISO format)
                if data.get("start_time"):
                    assert isinstance(data["start_time"], str)
                    # Verify it's a valid ISO format
                    datetime.fromisoformat(data["start_time"])


class TestPerformanceLogging:
    """Tests for performance logging functionality."""

    def test_performance_collector_records_metrics(self):
        """Test that performance collector records detailed metrics."""
        PerformanceCollector.reset()

        # Create a mock response with Ollama internal metrics
        mock_response = GenerateResponse(
            text="Hello!",
            model="qwen2.5vl:7b",
            total_duration=1000000000,  # 1 second in nanoseconds
            load_duration=100000000,  # 100ms
            prompt_eval_count=10,
            prompt_eval_duration=200000000,  # 200ms
            eval_count=20,
            eval_duration=500000000,  # 500ms
        )

        PerformanceCollector.record_performance(
            model="qwen2.5vl:7b",
            operation="generate",
            total_latency_ms=1000.0,
            success=True,
            response=mock_response,
        )

        stats = PerformanceCollector.get_performance_stats()
        assert stats["total_requests"] == 1
        assert stats["avg_tokens_per_second"] > 0
        assert "qwen2.5vl:7b" in stats["by_model"]

    def test_track_performance_context_manager(self):
        """Test track_performance context manager."""
        PerformanceCollector.reset()

        # The context manager needs response to be passed after the block completes
        # So we need to test this differently - just verify it records the request
        with track_performance("qwen2.5vl:7b", "generate"):
            time.sleep(0.01)

        # Verify a metric was recorded (even without detailed performance data)
        metrics = getattr(PerformanceCollector, "_metrics", [])
        assert len(metrics) == 1
        assert metrics[0].model == "qwen2.5vl:7b"
        assert metrics[0].total_latency_ms >= 10.0


class TestResilienceFeatures:
    """Tests for resilience features."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_proceed() is True

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker()

        # Record failures up to threshold
        for _ in range(5):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_proceed() is False

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker moves to HALF_OPEN after timeout."""
        cb = CircuitBreaker()

        # Force circuit to OPEN state
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate timeout by setting last_open_time to past
        cb.last_open_time = time.time() - 61  # Config timeout is 60s

        # Should transition to HALF_OPEN
        assert cb.can_proceed() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_exponential_backoff_retry_success(self):
        """Test exponential backoff retry with successful function."""
        call_count = [0]

        def successful_func():
            call_count[0] += 1
            return "success"

        result = exponential_backoff_retry(successful_func)
        assert result == "success"
        assert call_count[0] == 1

    def test_exponential_backoff_retry_with_retries(self):
        """Test exponential backoff retry with eventual success."""
        call_count = [0]

        def failing_then_success():
            call_count[0] += 1
            if call_count[0] < 3:
                raise requests.RequestException("Temporary failure")
            return "success"

        result = exponential_backoff_retry(failing_then_success)
        assert result == "success"
        assert call_count[0] == 3

    def test_exponential_backoff_retry_exhausted(self):
        """Test exponential backoff retry with exhausted retries."""

        def always_fails():
            raise requests.RequestException("Permanent failure")

        with pytest.raises(requests.RequestException):
            exponential_backoff_retry(always_fails)


class TestProjectTracking:
    """Tests for project-level analytics tracking."""

    def test_track_request_with_project(self):
        """Test tracking requests with project identifier."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        AnalyticsCollector.record_request_with_project(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=50.0,
            success=True,
            project="knowledge_machine",
        )

        analytics = AnalyticsCollector.get_analytics()
        assert analytics.total_requests == 1
        assert "knowledge_machine" in analytics.requests_by_project
        assert analytics.requests_by_project["knowledge_machine"] == 1

    def test_track_request_with_project_context_manager(self):
        """Test track_request_with_project context manager."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        with track_request_with_project(
            "qwen2.5vl:7b", "generate", project="test_project"
        ):
            time.sleep(0.01)

        analytics = AnalyticsCollector.get_analytics()
        assert analytics.total_requests == 1
        assert "test_project" in analytics.requests_by_project

    def test_filter_analytics_by_project(self):
        """Test filtering analytics by project."""
        MetricsCollector.reset()
        AnalyticsCollector._project_metadata.clear()

        # Record metrics for different projects
        AnalyticsCollector.record_request_with_project(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=50.0,
            success=True,
            project="project_a",
        )
        AnalyticsCollector.record_request_with_project(
            model="qwen2.5vl:7b",
            operation="generate",
            latency_ms=60.0,
            success=True,
            project="project_b",
        )

        # Get analytics for project_a only - verify project-level filtering
        analytics_a = AnalyticsCollector.get_analytics(project="project_a")
        # The project_metrics should only have project_a
        assert "project_a" in analytics_a.project_metrics
        assert analytics_a.project_metrics["project_a"].total_requests == 1

        # Get analytics for project_b only
        analytics_b = AnalyticsCollector.get_analytics(project="project_b")
        assert "project_b" in analytics_b.project_metrics
        assert analytics_b.project_metrics["project_b"].total_requests == 1


class TestTypeHints:
    """Tests for type hint improvements."""

    def test_generator_type_hints_track_request(self):
        """Test that track_request returns a context manager."""
        from contextlib import _GeneratorContextManager

        MetricsCollector.reset()
        cm = track_request("qwen2.5vl:7b", "generate")

        # Context managers from @contextmanager are _GeneratorContextManager
        assert isinstance(cm, _GeneratorContextManager)

    def test_generator_type_hints_track_request_with_project(self):
        """Test that track_request_with_project returns a context manager."""
        from contextlib import _GeneratorContextManager

        MetricsCollector.reset()
        cm = track_request_with_project("qwen2.5vl:7b", "generate", project="test")

        # Context managers from @contextmanager are _GeneratorContextManager
        assert isinstance(cm, _GeneratorContextManager)


class TestErrorPreservation:
    """Tests for error type preservation."""

    def test_error_type_preserved_in_metrics(self):
        """Test that error type is preserved in error messages."""
        MetricsCollector.reset()

        # Simulate an error during tracking
        try:
            with track_request("qwen2.5vl:7b", "generate"):
                raise ValueError("Test error message")
        except ValueError:
            pass

        # Verify error type is preserved in metrics
        metrics = getattr(MetricsCollector, "_metrics", [])
        assert len(metrics) > 0
        assert metrics[0].error is not None
        assert "ValueError" in metrics[0].error
        assert "Test error message" in metrics[0].error
