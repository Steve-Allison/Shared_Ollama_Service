import argparse
import asyncio
import io
import json
from contextlib import redirect_stdout

from shared_ollama import (
    AnalyticsCollector,
    MetricsCollector,
    track_request,
    track_request_with_project,
)

from scripts.async_load_test import run_load_test
from scripts.performance_report import calculate_performance_stats, parse_performance_log
from scripts.view_analytics import print_analytics_dashboard


def test_async_load_test_cli_smoke(ollama_server):
    args = argparse.Namespace(
        base_url=ollama_server.base_url,
        model="qwen2.5vl:7b",
        prompt="Load test prompt",
        workers=2,
        concurrency=None,
        requests=6,
        duration=0.0,
        delay=0.0,
        warmup=1,
        max_connections=None,
        keepalive=None,
        timeout=30.0,
        health_timeout=5.0,
        sample_count=10,
        output=None,
        skip_verify=False,
    )

    report = asyncio.run(run_load_test(args))
    summary = report["summary"]
    assert summary["total_requests"] == 6
    assert summary["successes"] == summary["total_requests"]


def test_view_analytics_dashboard_outputs(ollama_server):
    MetricsCollector.reset()
    AnalyticsCollector._project_metadata.clear()

    with track_request("qwen2.5vl:7b", "generate"):
        pass
    with track_request_with_project("qwen2.5vl:7b", "generate", project="proj-cli"):
        pass

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_analytics_dashboard()

    output = buffer.getvalue()
    assert "Shared Ollama Service - Analytics Dashboard" in output
    assert "Total Requests" in output


def test_performance_report_parsing(tmp_path):
    log_file = tmp_path / "perf.jsonl"
    entry = {
        "timestamp": "2025-11-12T12:00:00Z",
        "model": "qwen2.5vl:7b",
        "success": True,
        "tokens_per_second": 120.0,
        "load_time_ms": 150.0,
        "generation_time_ms": 320.0,
    }
    log_file.write_text(json.dumps(entry) + "\n")

    metrics = parse_performance_log(log_file)
    stats = calculate_performance_stats(metrics)

    assert stats["total_requests"] == 1
    assert stats["successful_requests"] == 1
    assert stats["avg_tokens_per_second"] == entry["tokens_per_second"]

