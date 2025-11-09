"""
Enhanced Performance Logging for Ollama
========================================

Captures and logs Ollama's internal performance metrics including:
- Model loading time
- Token generation speed
- Prompt evaluation performance
- Detailed performance breakdown

Usage:
    from performance_logging import track_performance, get_performance_stats

    # Track with detailed metrics (works with any model: qwen2.5vl:7b, qwen2.5:7b, qwen2.5:14b, granite4:tiny-h)
    with track_performance("qwen2.5vl:7b", "generate"):
        response = client.generate("Hello!")

    # Get performance statistics
    stats = get_performance_stats()
    print(f"Avg tokens/sec: {stats.avg_tokens_per_second}")
"""

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from shared_ollama_client import GenerateResponse

# Setup structured performance logging
performance_logger = logging.getLogger("ollama.performance")
performance_logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

# JSON Lines file handler for structured logs
performance_handler = logging.FileHandler(logs_dir / "performance.jsonl")
performance_handler.setFormatter(logging.Formatter("%(message)s"))
performance_logger.addHandler(performance_handler)


@dataclass
class DetailedPerformanceMetrics:
    """Detailed performance metrics including Ollama internal data."""

    model: str
    operation: str
    timestamp: datetime

    # Overall metrics
    total_latency_ms: float
    success: bool
    error: str | None = None

    # Ollama internal metrics (from GenerateResponse)
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None

    # Calculated metrics
    load_time_ms: float | None = None
    prompt_eval_time_ms: float | None = None
    generation_time_ms: float | None = None
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "operation": self.operation,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "success": self.success,
            "error": self.error,
            "load_time_ms": round(self.load_time_ms, 2) if self.load_time_ms else None,
            "prompt_eval_time_ms": round(self.prompt_eval_time_ms, 2)
            if self.prompt_eval_time_ms
            else None,
            "generation_time_ms": round(self.generation_time_ms, 2)
            if self.generation_time_ms
            else None,
            "prompt_tokens": self.prompt_eval_count,
            "generated_tokens": self.eval_count,
            "tokens_per_second": round(self.tokens_per_second, 2)
            if self.tokens_per_second
            else None,
            "prompt_tokens_per_second": round(self.prompt_tokens_per_second, 2)
            if self.prompt_tokens_per_second
            else None,
        }


class PerformanceCollector:
    """Collects detailed performance metrics for Ollama requests."""

    _metrics: ClassVar[list[DetailedPerformanceMetrics]] = []
    _max_metrics: ClassVar[int] = 10000  # Keep last 10k requests

    @classmethod
    def record_performance(
        cls,
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: GenerateResponse | None = None,
        error: str | None = None,
    ):
        """
        Record detailed performance metrics.

        Args:
            model: Model name used
            operation: Operation type
            total_latency_ms: Total request latency
            success: Whether request succeeded
            response: GenerateResponse with Ollama internal metrics
            error: Error message if failed
        """
        metric = DetailedPerformanceMetrics(
            model=model,
            operation=operation,
            timestamp=datetime.now(UTC),
            total_latency_ms=total_latency_ms,
            success=success,
            error=error,
        )

        # Extract Ollama internal metrics if available
        if response and success:
            metric.load_duration_ns = response.load_duration
            metric.prompt_eval_count = response.prompt_eval_count
            metric.prompt_eval_duration_ns = response.prompt_eval_duration
            metric.eval_count = response.eval_count
            metric.eval_duration_ns = response.eval_duration
            metric.total_duration_ns = response.total_duration

            # Calculate derived metrics
            if metric.load_duration_ns:
                metric.load_time_ms = metric.load_duration_ns / 1_000_000

            if metric.prompt_eval_duration_ns:
                metric.prompt_eval_time_ms = metric.prompt_eval_duration_ns / 1_000_000

            if metric.eval_duration_ns:
                metric.generation_time_ms = metric.eval_duration_ns / 1_000_000

            # Calculate tokens per second
            if metric.eval_duration_ns and metric.eval_count:
                # Convert nanoseconds to seconds
                eval_seconds = metric.eval_duration_ns / 1_000_000_000
                if eval_seconds > 0:
                    metric.tokens_per_second = metric.eval_count / eval_seconds

            if metric.prompt_eval_duration_ns and metric.prompt_eval_count:
                prompt_seconds = metric.prompt_eval_duration_ns / 1_000_000_000
                if prompt_seconds > 0:
                    metric.prompt_tokens_per_second = metric.prompt_eval_count / prompt_seconds

        # Store metric
        cls._metrics.append(metric)

        # Prune old metrics
        if len(cls._metrics) > cls._max_metrics:
            cls._metrics = cls._metrics[-cls._max_metrics :]

        # Log to JSON Lines file
        performance_logger.info(json.dumps(metric.to_dict()))

        # Also log summary to standard logger
        if metric.tokens_per_second:
            logging.info(
                f"Performance: {model} - {metric.tokens_per_second:.1f} tokens/sec, "
                f"load: {metric.load_time_ms:.1f}ms, gen: {metric.generation_time_ms:.1f}ms"
            )

    @classmethod
    def get_performance_stats(cls) -> dict[str, Any]:
        """Get aggregated performance statistics."""
        if not cls._metrics:
            return {}

        successful = [m for m in cls._metrics if m.success and m.tokens_per_second]

        if not successful:
            return {}

        # Aggregate statistics
        tokens_per_second_values = [
            m.tokens_per_second for m in successful if m.tokens_per_second is not None
        ]
        avg_tokens_per_second = (
            sum(tokens_per_second_values) / len(tokens_per_second_values)
            if tokens_per_second_values
            else 0.0
        )

        load_times = [m.load_time_ms for m in successful if m.load_time_ms is not None]
        avg_load_time = sum(load_times) / len(load_times) if load_times else 0.0

        generation_times = [
            m.generation_time_ms for m in successful if m.generation_time_ms is not None
        ]
        avg_generation_time = (
            sum(generation_times) / len(generation_times) if generation_times else 0.0
        )

        # By model
        by_model: dict[str, list[DetailedPerformanceMetrics]] = defaultdict(list)
        for m in successful:
            by_model[m.model].append(m)

        model_stats = {}
        for model, metrics in by_model.items():
            model_stats[model] = {
                "avg_tokens_per_second": (
                    sum(m.tokens_per_second for m in metrics if m.tokens_per_second is not None)
                    / len([m for m in metrics if m.tokens_per_second is not None])
                    if any(m.tokens_per_second for m in metrics)
                    else 0.0
                ),
                "avg_load_time_ms": (
                    sum(m.load_time_ms for m in metrics if m.load_time_ms is not None)
                    / len([m for m in metrics if m.load_time_ms is not None])
                    if any(m.load_time_ms for m in metrics)
                    else 0.0
                )
                if any(m.load_time_ms for m in metrics)
                else 0.0,
                "avg_generation_time_ms": (
                    sum(m.generation_time_ms for m in metrics if m.generation_time_ms is not None)
                    / len([m for m in metrics if m.generation_time_ms is not None])
                    if any(m.generation_time_ms for m in metrics)
                    else 0.0
                ),
                "request_count": len(metrics),
            }

        return {
            "avg_tokens_per_second": round(avg_tokens_per_second, 2),
            "avg_load_time_ms": round(avg_load_time, 2) if avg_load_time else 0,
            "avg_generation_time_ms": round(avg_generation_time, 2) if avg_generation_time else 0,
            "total_requests": len(successful),
            "by_model": model_stats,
        }

    @classmethod
    def reset(cls):
        """Reset all metrics (for testing)."""
        cls._metrics = []


@contextmanager
def track_performance(
    model: str,
    operation: str = "generate",
    response: GenerateResponse | None = None,
):
    """
    Context manager to track request with detailed Ollama performance metrics.

    Args:
        model: Model name
        operation: Operation type
        response: GenerateResponse object (optional, for detailed metrics)

    Example:
        >>> response = client.generate("Hello!")
        >>> with track_performance("qwen2.5vl:7b", "generate", response=response):
        ...     pass  # Already executed
    """
    start_time = time.perf_counter()
    success = False
    error = None

    try:
        yield
        success = True
    except Exception as e:
        error = str(e)
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
    """Get aggregated performance statistics."""
    return PerformanceCollector.get_performance_stats()
