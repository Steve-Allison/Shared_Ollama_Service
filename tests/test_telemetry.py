import json
from pathlib import Path
import logging

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


def test_track_request_success_and_failure():
    MetricsCollector.reset()

    with track_request("qwen2.5vl:7b", "generate"):
        pass

    with pytest.raises(ValueError):
        with track_request("qwen2.5vl:7b", "generate"):
            raise ValueError("boom")

    metrics = MetricsCollector.get_metrics()
    assert isinstance(metrics, ServiceMetrics)
    assert metrics.total_requests == 2
    assert metrics.successful_requests == 1
    assert metrics.failed_requests == 1
    assert "ValueError: boom" in MetricsCollector._metrics[-1].error  # type: ignore[attr-defined]


def test_analytics_collection_and_exports(tmp_path):
    MetricsCollector.reset()
    AnalyticsCollector._project_metadata.clear()

    with track_request_with_project("qwen2.5vl:7b", "generate", project="proj-A"):
        pass

    report = AnalyticsCollector.get_analytics()
    assert report.requests_by_project["proj-A"] == 1
    assert isinstance(report.project_metrics["proj-A"], ProjectMetrics)

    json_path = tmp_path / "analytics.json"
    csv_path = tmp_path / "analytics.csv"

    AnalyticsCollector.export_json(json_path)
    AnalyticsCollector.export_csv(csv_path)

    assert json_path.exists()
    assert csv_path.exists()

    exported = json.loads(json_path.read_text())
    assert exported["total_requests"] == 1


def test_performance_tracking_updates_stats():
    PerformanceCollector.reset()
    logs_path = Path(__file__).resolve().parents[1] / "logs" / "performance.jsonl"
    before = logs_path.read_text().splitlines() if logs_path.exists() else []

    response = GenerateResponse(
        text="done",
        model="qwen2.5vl:7b",
        context=[1, 2],
        total_duration=400_000_000,
        load_duration=100_000_000,
        prompt_eval_count=4,
        prompt_eval_duration=50_000_000,
        eval_count=10,
        eval_duration=200_000_000,
    )

    with track_performance("qwen2.5vl:7b", response=response):
        pass

    stats = get_performance_stats()
    assert stats["total_requests"] == 1
    assert stats["avg_tokens_per_second"] > 0

    after = logs_path.read_text().splitlines()
    assert len(after) == len(before) + 1


def test_structured_logging_appends_json(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    # Temporarily redirect the logger to our temp file
    original_handlers = []
    from shared_ollama.telemetry import structured_logging  # noqa: WPS433

    for handler in structured_logging.REQUEST_LOGGER.handlers:
        original_handlers.append(handler)
    structured_logging.REQUEST_LOGGER.handlers = []
    new_handler = logging.FileHandler(log_path)
    structured_logging.REQUEST_LOGGER.addHandler(new_handler)

    try:
        log_request_event({"event": "test", "model": "qwen"})
        data = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert data[-1]["event"] == "test"
        assert data[-1]["model"] == "qwen"
    finally:
        structured_logging.REQUEST_LOGGER.removeHandler(new_handler)
        new_handler.close()
        structured_logging.REQUEST_LOGGER.handlers = original_handlers

