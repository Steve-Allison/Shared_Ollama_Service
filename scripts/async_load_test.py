#!/usr/bin/env python3
"""
Async load-testing harness for the Shared Ollama Service.

This script exercises the async client with configurable concurrency and
captures latency metrics plus error summaries for post-run tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared_ollama.client import AsyncOllamaConfig, AsyncSharedOllamaClient  # noqa: E402

DEFAULT_VLM_MODEL = os.getenv("OLLAMA_DEFAULT_VLM_MODEL", "qwen3-vl:8b-instruct-q4_K_M")


def percentile(values: list[float], pct: float) -> float:
    """Return the percentile (0-1) value for a list of floats.

    Uses statistics.quantiles() for consistency with production code.
    """
    import statistics

    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 1:
        return max(values)

    # Use statistics.quantiles with method='inclusive' for consistent behavior
    quantiles = statistics.quantiles(values, n=100, method='inclusive')
    index = min(int(pct * 100), len(quantiles) - 1)
    return quantiles[index]


async def warmup(client: AsyncSharedOllamaClient, prompt: str, model: str, count: int) -> None:
    """Run a few warm-up requests to load models before measuring."""
    if count <= 0:
        return

    for i in range(count):
        try:
            await client.generate(prompt, model=model, stream=False)
        except Exception as exc:
            print(f"[warmup {i+1}/{count}] failed: {exc}")
            break


async def run_load_test(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the load test and return the structured report."""
    max_connections = args.max_connections or max(args.workers * 2, 50)
    max_keepalive = args.keepalive or max(int(max_connections / 2), 10)
    max_concurrency = args.concurrency or args.workers

    config = AsyncOllamaConfig(
        base_url=args.base_url,
        default_model=args.model,
        timeout=args.timeout,
        health_check_timeout=args.health_timeout,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
        max_concurrent_requests=max_concurrency,
    )

    queue_mode = args.duration <= 0
    total_requests = args.requests if queue_mode else 0
    duration_seconds = args.duration if args.duration > 0 else None

    # Shared state
    request_counter = count()
    durations: list[float] = []
    success_count = 0
    failure_count = 0
    error_counter: Counter[str] = Counter()
    sample_requests: list[dict[str, Any]] = []
    sample_errors: list[dict[str, Any]] = []

    queue: asyncio.Queue[int] | None = None
    if queue_mode:
        queue = asyncio.Queue()
        for idx in range(total_requests):
            queue.put_nowait(idx)

    stop_event = asyncio.Event()
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds if duration_seconds else None

    async with AsyncSharedOllamaClient(config=config, verify_on_init=not args.skip_verify) as client:
        await warmup(client, args.prompt, args.model, args.warmup)

        async def worker(worker_id: int) -> None:
            nonlocal success_count, failure_count

            def time_exceeded() -> bool:
                if not end_time:
                    return False
                if time.perf_counter() < end_time:
                    return False
                stop_event.set()
                return True

            def next_request_id() -> int | None:
                if queue_mode:
                    assert queue is not None
                    try:
                        return queue.get_nowait()
                    except asyncio.QueueEmpty:
                        stop_event.set()
                        return None
                return next(request_counter)

            def record_sample(entry: dict[str, Any], error_present: bool) -> None:
                if len(sample_requests) < args.sample_count:
                    sample_requests.append(entry)
                if error_present and len(sample_errors) < args.sample_count:
                    sample_errors.append(entry)

            while not stop_event.is_set():
                if time_exceeded():
                    break

                request_id = next_request_id()
                if request_id is None:
                    break

                t0 = time.perf_counter()
                error_message: str | None = None
                success = True
                try:
                    await client.generate(args.prompt, model=args.model, stream=False)
                except Exception as exc:
                    success = False
                    error_message = str(exc)
                    error_counter[exc.__class__.__name__] += 1
                duration = time.perf_counter() - t0

                durations.append(duration)
                success_count += int(success)
                failure_count += int(not success)

                entry = {
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "success": success,
                    "duration_ms": duration * 1000,
                }
                if error_message:
                    entry["error"] = error_message

                record_sample(entry, bool(error_message))

                if queue_mode and queue is not None:
                    queue.task_done()

                if time_exceeded():
                    break

                if args.delay > 0:
                    await asyncio.sleep(args.delay)

        workers = [asyncio.create_task(worker(i)) for i in range(args.workers)]
        await asyncio.gather(*workers)

        if queue_mode and queue is not None:
            await queue.join()

    total_elapsed = time.perf_counter() - start_time
    request_total = len(durations)
    rps = request_total / total_elapsed if total_elapsed > 0 else 0.0

    summary = {
        "total_requests": request_total,
        "successes": success_count,
        "failures": failure_count,
        "success_rate": (success_count / request_total) * 100 if request_total else 0.0,
        "test_duration_seconds": total_elapsed,
        "requests_per_second": rps,
        "latency_ms": {
            "min": min(durations) * 1000 if durations else 0.0,
            "max": max(durations) * 1000 if durations else 0.0,
            "avg": (sum(durations) / request_total) * 1000 if request_total else 0.0,
            "p50": percentile(durations, 0.5) * 1000,
            "p95": percentile(durations, 0.95) * 1000,
            "p99": percentile(durations, 0.99) * 1000,
        },
        "errors": error_counter,
    }

    report = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt": args.prompt,
            "workers": args.workers,
            "concurrency": max_concurrency,
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive,
            "requests": args.requests if queue_mode else None,
            "duration": args.duration if args.duration > 0 else None,
            "warmup": args.warmup,
            "delay": args.delay,
        },
        "summary": summary,
        "samples": {
            "requests": sample_requests,
            "errors": sample_errors,
        },
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async load tester for the Shared Ollama Service.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama service base URL.")
    parser.add_argument("--model", default=DEFAULT_VLM_MODEL, help="Model to exercise during the test.")
    parser.add_argument(
        "--prompt",
        default="Summarize the concept of vector databases in one short paragraph.",
        help="Prompt to send with each request.",
    )
    parser.add_argument("--workers", type=int, default=10, help="Number of asyncio workers issuing requests.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent requests (defaults to number of workers).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=200,
        help="Total requests to send (ignored when --duration > 0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Test duration in seconds. When > 0 overrides --requests.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between successive requests per worker.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Number of warm-up requests before the test.")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Override httpx max_connections (defaults to workers*2 or 50, whichever is larger).",
    )
    parser.add_argument(
        "--keepalive",
        type=int,
        default=None,
        help="Override httpx max_keepalive_connections (defaults to half of max-connections).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Read timeout passed to the async client (seconds).",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=5.0,
        help="Health check timeout for /api/tags (seconds).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Maximum number of individual request samples to include in the report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report (defaults to logs/perf_reports/<timestamp>.json).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip initial connectivity verification on client creation.",
    )
    return parser.parse_args()


def save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nReport written to: {output_path}")


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("\n=== Async Load Test Summary ===")
    print(f"Total requests:    {summary['total_requests']}")
    print(f"Successes:         {summary['successes']}")
    print(f"Failures:          {summary['failures']}")
    print(f"Success rate:      {summary['success_rate']:.2f}%")
    print(f"Test duration (s): {summary['test_duration_seconds']:.2f}")
    print(f"Throughput (RPS):  {summary['requests_per_second']:.2f}")
    print("Latency (ms):")
    for key, value in summary["latency_ms"].items():
        print(f"  {key:<4}: {value:.2f}")
    if summary["errors"]:
        print("Errors:")
        for err, count in summary["errors"].items():
            print(f"  {err}: {count}")


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_load_test(args))

    output_path = (
        args.output
        if args.output is not None
        else PROJECT_ROOT / "logs" / "perf_reports" / f"async_load_test_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    )

    print_summary(report)
    save_report(report, output_path)


if __name__ == "__main__":
    main()

