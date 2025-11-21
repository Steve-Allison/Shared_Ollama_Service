"""Type stubs for shared_ollama.telemetry.performance module."""

from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

from shared_ollama.client.sync import GenerateResponse

class DetailedPerformanceMetrics:
    model: str
    operation: str
    timestamp: datetime
    total_latency_ms: float
    success: bool
    error: str | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    prompt_eval_duration_ns: int | None
    eval_count: int | None
    eval_duration_ns: int | None
    total_duration_ns: int | None
    load_time_ms: float | None
    prompt_eval_time_ms: float | None
    generation_time_ms: float | None
    tokens_per_second: float | None
    prompt_tokens_per_second: float | None

    def to_dict(self) -> dict[str, Any]: ...

class PerformanceCollector:
    @classmethod
    def record_performance(
        cls,
        model: str,
        operation: str,
        total_latency_ms: float,
        success: bool,
        response: GenerateResponse | dict[str, Any] | None = ...,
        error: str | None = ...,
    ) -> None: ...
    @classmethod
    def get_performance_stats(cls) -> dict[str, Any]: ...
    @classmethod
    def reset(cls) -> None: ...

def track_performance(
    model: str,
    operation: str = ...,
    response: GenerateResponse | None = ...,
) -> AbstractContextManager[None]: ...

def get_performance_stats() -> dict[str, Any]: ...
