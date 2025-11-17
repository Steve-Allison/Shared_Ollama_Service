"""Performance logging utilities for the Shared Ollama Service.

This module provides detailed performance metrics collection including
token generation rates, model load times, and evaluation durations.

Key behaviors:
    - Detailed timing metrics from Ollama responses (nanoseconds)
    - Automatic conversion to milliseconds and tokens/second
    - JSON Lines logging to performance.jsonl
    - In-memory storage with automatic size limiting
    - Model-specific performance aggregation

Log file:
    - Location: ``logs/performance.jsonl`` (relative to project root)
    - Format: One JSON object per line
    - Encoding: UTF-8
"""

from __future__ import annotations

import functools
import json
import logging
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    from shared_ollama.client.sync import GenerateResponse

performance_logger = logging.getLogger("ollama.performance")
performance_logger.setLevel(logging.INFO)

# Cache log directory resolution
@functools.cache
def _get_logs_dir() -> Path:
    """Get logs directory with caching for performance.

    Resolves the logs directory relative to the project root and creates
    it if it doesn't exist. Result is cached since the path doesn't change
    at runtime.

    Returns:
        Path to logs directory.

    Side effects:
        Creates logs directory if it doesn't exist.
    """
    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


LOGS_DIR = _get_logs_dir()

performance_handler = logging.FileHandler(LOGS_DIR / "performance.jsonl")
performance_handler.setFormatter(logging.Formatter("%(message)s"))
performance_logger.addHandler(performance_handler)


@dataclass(slots=True)
class DetailedPerformanceMetrics:
    """Detailed performance metrics for a request.

    Contains comprehensive timing and token metrics extracted from Ollama
    responses. All duration values are converted from nanoseconds to
    milliseconds for readability.

    Attributes:
        model: Model name used for the request.
        operation: Operation type (e.g., "generate", "chat").
        timestamp: Request timestamp in UTC.
        total_latency_ms: Total request latency in milliseconds (>=0.0).
        success: Whether the request succeeded.
        error: Error message if request failed. None if successful.
        load_duration_ns: Model load duration in nanoseconds. None if N/A.
        prompt_eval_count: Number of prompt tokens evaluated. None if N/A.
        prompt_eval_duration_ns: Prompt evaluation duration in nanoseconds. None if N/A.
        eval_count: Number of generation tokens produced. None if N/A.
        eval_duration_ns: Generation duration in nanoseconds. None if N/A.
        total_duration_ns: Total generation duration in nanoseconds. None if N/A.
        load_time_ms: Model load time in milliseconds (converted from ns). None if N/A.
        prompt_eval_time_ms: Prompt evaluation time in milliseconds. None if N/A.
        generation_time_ms: Generation time in milliseconds. None if N/A.
        tokens_per_second: Token generation rate (tokens/sec). None if N/A.
        prompt_tokens_per_second: Prompt evaluation rate (tokens/sec). None if N/A.
    """

    model: str
    operation: str
    timestamp: datetime
    total_latency_ms: float
    success: bool
    error: str | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    load_time_ms: float | None = None
    prompt_eval_time_ms: float | None = None
    generation_time_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization.

        Converts all fields to JSON-serializable format. Timestamps are
        converted to ISO 8601 strings. Numeric values are rounded to
        2 decimal places.

        Returns:
            Dictionary with all metrics in JSON-serializable format.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "operation": self.operation,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "success": self.success,
            "error": self.error,
            "load_time_ms": round(self.load_time_ms, 2) if self.load_time_ms else None,
            "prompt_eval_time_ms": round(self.prompt_eval_time_ms, 2) if self.prompt_eval_time_ms else None,
            "generation_time_ms": round(self.generation_time_ms, 2) if self.generation_time_ms else None,
            "prompt_tokens": self.prompt_eval_count,
            "generated_tokens": self.eval_count,
            "tokens_per_second": round(self.tokens_per_second, 2) if self.tokens_per_second else None,
            "prompt_tokens_per_second": round(self.prompt_tokens_per_second, 2)
            if self.prompt_tokens_per_second
            else None,
        }


class PerformanceCollector:
    """Collects detailed performance metrics for requests.

    Extracts detailed timing and token metrics from Ollama responses and
    computes derived metrics like tokens/second. Stores metrics in-memory
    and logs to JSON Lines file.

    Attributes:
        _metrics: Class variable storing list of DetailedPerformanceMetrics.
        _max_metrics: Maximum number of metrics to retain (default: 10,000).

    Thread safety:
        Not thread-safe. Use from a single thread or protect with locks
        if accessing from multiple threads.

    Memory management:
        Automatically trims oldest metrics when _max_metrics is exceeded.
    """

    _metrics: ClassVar[list[DetailedPerformanceMetrics]] = []
    _max_metrics: ClassVar[int] = 10_000

    @classmethod
    def record_performance(
        cls,
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: GenerateResponse | dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Record detailed performance metrics.

        Extracts timing and token metrics from GenerateResponse or dict and computes
        derived metrics. Logs to performance.jsonl file.

        Args:
            model: Model name used for the request.
            operation: Operation type (e.g., "generate", "chat").
            total_latency_ms: Total request latency in milliseconds (>=0.0).
            success: Whether the request succeeded.
            response: GenerateResponse object or dict with detailed timing data.
                Dict format should have keys: load_duration, prompt_eval_count,
                prompt_eval_duration, eval_count, eval_duration, total_duration.
                Only used if success is True.
            error: Error message if request failed. None if successful.

        Side effects:
            - Creates DetailedPerformanceMetrics object
            - Appends to _metrics list (may trim if limit exceeded)
            - Writes JSON line to performance.jsonl file
            - Logs info message with tokens/second if available
        """
        metric = DetailedPerformanceMetrics(
            model=model,
            operation=operation,
            timestamp=datetime.now(UTC),
            total_latency_ms=total_latency_ms,
            success=success,
            error=error,
        )

        if response and success:
            # Handle both GenerateResponse objects and dicts
            if isinstance(response, dict):
                # Extract from dict (adapter returns dicts)
                metric.load_duration_ns = response.get("load_duration")
                metric.prompt_eval_count = response.get("prompt_eval_count")
                metric.prompt_eval_duration_ns = response.get("prompt_eval_duration")
                metric.eval_count = response.get("eval_count")
                metric.eval_duration_ns = response.get("eval_duration")
                metric.total_duration_ns = response.get("total_duration")
            else:
                # Handle GenerateResponse object (backward compatibility)
                metric.load_duration_ns = response.load_duration
                metric.prompt_eval_count = response.prompt_eval_count
                metric.prompt_eval_duration_ns = response.prompt_eval_duration
                metric.eval_count = response.eval_count
                metric.eval_duration_ns = response.eval_duration
                metric.total_duration_ns = response.total_duration

            if metric.load_duration_ns:
                metric.load_time_ms = metric.load_duration_ns / 1_000_000

            if metric.prompt_eval_duration_ns:
                metric.prompt_eval_time_ms = metric.prompt_eval_duration_ns / 1_000_000

            if metric.eval_duration_ns:
                metric.generation_time_ms = metric.eval_duration_ns / 1_000_000

            if metric.eval_duration_ns and metric.eval_count:
                eval_seconds = metric.eval_duration_ns / 1_000_000_000
                if eval_seconds > 0:
                    metric.tokens_per_second = metric.eval_count / eval_seconds

            if metric.prompt_eval_duration_ns and metric.prompt_eval_count:
                prompt_seconds = metric.prompt_eval_duration_ns / 1_000_000_000
                if prompt_seconds > 0:
                    metric.prompt_tokens_per_second = metric.prompt_eval_count / prompt_seconds

        cls._metrics.append(metric)

        if len(cls._metrics) > cls._max_metrics:
            cls._metrics = cls._metrics[-cls._max_metrics :]

        performance_logger.info(json.dumps(metric.to_dict()))

        if metric.tokens_per_second:
            logging.info(
                "Performance: %s - %.1f tokens/sec, load: %.1fms, gen: %.1fms",
                model,
                metric.tokens_per_second,
                metric.load_time_ms or 0.0,
                metric.generation_time_ms or 0.0,
            )

    @classmethod
    def get_performance_stats(cls) -> dict[str, Any]:
        """Get aggregated performance statistics.

        Computes average performance metrics across all successful requests
        with token generation data. Groups statistics by model.

        Returns:
            Dictionary with keys:
                - avg_tokens_per_second: float - Overall average
                - avg_load_time_ms: float - Overall average
                - avg_generation_time_ms: float - Overall average
                - total_requests: int - Count of successful requests
                - by_model: dict[str, dict] - Per-model statistics with same
                  structure as top-level keys plus request_count

            Returns empty dict if no successful metrics available.
        """
        if not cls._metrics:
            return {}

        successful = [
            metric for metric in cls._metrics if metric.success and metric.tokens_per_second
        ]
        if not successful:
            return {}

        tokens_per_second_values = [
            metric.tokens_per_second for metric in successful if metric.tokens_per_second is not None
        ]
        avg_tokens_per_second = (
            sum(tokens_per_second_values) / len(tokens_per_second_values) if tokens_per_second_values else 0.0
        )

        load_times = [metric.load_time_ms for metric in successful if metric.load_time_ms is not None]
        avg_load_time = sum(load_times) / len(load_times) if load_times else 0.0

        generation_times = [
            metric.generation_time_ms for metric in successful if metric.generation_time_ms is not None
        ]
        avg_generation_time = (
            sum(generation_times) / len(generation_times) if generation_times else 0.0
        )

        by_model: defaultdict[str, list[DetailedPerformanceMetrics]] = defaultdict(list)
        for metric in successful:
            by_model[metric.model].append(metric)

        model_stats: dict[str, dict[str, Any]] = {}
        for model, metrics in by_model.items():
            tokens = [m.tokens_per_second for m in metrics if m.tokens_per_second is not None]
            loads = [m.load_time_ms for m in metrics if m.load_time_ms is not None]
            generations = [m.generation_time_ms for m in metrics if m.generation_time_ms is not None]

            model_stats[model] = {
                "avg_tokens_per_second": sum(tokens) / len(tokens) if tokens else 0.0,
                "avg_load_time_ms": sum(loads) / len(loads) if loads else 0.0,
                "avg_generation_time_ms": sum(generations) / len(generations) if generations else 0.0,
                "request_count": len(metrics),
            }

        return {
            "avg_tokens_per_second": round(avg_tokens_per_second, 2),
            "avg_load_time_ms": round(avg_load_time, 2) if avg_load_time else 0.0,
            "avg_generation_time_ms": round(avg_generation_time, 2) if avg_generation_time else 0.0,
            "total_requests": len(successful),
            "by_model": model_stats,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset all collected performance metrics.

        Clears the entire metrics collection. Useful for testing or
        periodic resets.

        Side effects:
            Sets _metrics to empty list.
        """
        cls._metrics = []


@contextmanager
def track_performance(
    model: str,
    operation: str = "generate",
    response: GenerateResponse | None = None,
) -> Generator[None, None, None]:
    """Context manager to track request performance.

    Automatically measures execution time and records detailed performance
    metrics. Handles exceptions by recording error information.

    Args:
        model: Model name for the request.
        operation: Operation type (e.g., "generate", "chat"). Defaults to "generate".
        response: GenerateResponse with detailed timing data. Should be set
            after the request completes if available.

    Yields:
        None. The context manager tracks timing while the context is active.

    Side effects:
        - Measures execution time using time.perf_counter()
        - Records metrics via PerformanceCollector.record_performance()
        - Logs error information if exception occurs

    Example:
        >>> response = None
        >>> with track_performance("qwen2.5vl:7b", "generate"):
        ...     response = await client.generate("Hello")
        >>> # Note: response should be passed to record_performance separately
    """
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
        PerformanceCollector.record_performance(
            model=model,
            operation=operation,
            total_latency_ms=latency_ms,
            success=success,
            response=response,
            error=error,
        )


def get_performance_stats() -> dict[str, Any]:
    """Get aggregated performance statistics.

    Convenience function that delegates to PerformanceCollector.get_performance_stats().

    Returns:
        Dictionary with performance statistics. See PerformanceCollector.get_performance_stats()
        for structure.
    """
    return PerformanceCollector.get_performance_stats()


__all__ = [
    "DetailedPerformanceMetrics",
    "PerformanceCollector",
    "get_performance_stats",
    "track_performance",
]
