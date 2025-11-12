"""
Performance logging utilities for the Shared Ollama Service.
"""

from __future__ import annotations

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

LOGS_DIR = Path(__file__).resolve().parents[3] / "logs"
LOGS_DIR.mkdir(exist_ok=True)

performance_handler = logging.FileHandler(LOGS_DIR / "performance.jsonl")
performance_handler.setFormatter(logging.Formatter("%(message)s"))
performance_logger.addHandler(performance_handler)


@dataclass
class DetailedPerformanceMetrics:
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
    _metrics: ClassVar[list[DetailedPerformanceMetrics]] = []
    _max_metrics: ClassVar[int] = 10_000

    @classmethod
    def record_performance(
        cls,
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: "GenerateResponse | None" = None,
        error: str | None = None,
    ) -> None:
        metric = DetailedPerformanceMetrics(
            model=model,
            operation=operation,
            timestamp=datetime.now(UTC),
            total_latency_ms=total_latency_ms,
            success=success,
            error=error,
        )

        if response and success:
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
        if not cls._metrics:
            return {}

        successful = [metric for metric in cls._metrics if metric.success and metric.tokens_per_second]
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

        by_model: dict[str, list[DetailedPerformanceMetrics]] = defaultdict(list)
        for metric in successful:
            by_model[metric.model].append(metric)

        model_stats = {}
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
        cls._metrics = []


@contextmanager
def track_performance(
    model: str,
    operation: str = "generate",
    response: "GenerateResponse | None" = None,
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
        PerformanceCollector.record_performance(
            model=model,
            operation=operation,
            total_latency_ms=latency_ms,
            success=success,
            response=response,
            error=error,
        )


def get_performance_stats() -> dict[str, Any]:
    return PerformanceCollector.get_performance_stats()


__all__ = [
    "DetailedPerformanceMetrics",
    "PerformanceCollector",
    "get_performance_stats",
    "track_performance",
]

