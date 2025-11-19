#!/usr/bin/env python3
"""
Performance Report Generator
============================

Analyzes performance logs and generates detailed performance reports.

Usage:
    python scripts/performance_report.py
    python scripts/performance_report.py --model qwen3-vl:32b
    python scripts/performance_report.py --window 60
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_performance_log(log_file: Path) -> list[dict]:
    """Parse performance.jsonl file."""
    metrics = []

    if not log_file.exists():
        return metrics

    with log_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return metrics


def calculate_performance_stats(metrics: list[dict]) -> dict:
    """Calculate performance statistics."""
    if not metrics:
        return {}

    successful = [m for m in metrics if m.get("success") and m.get("tokens_per_second")]

    if not successful:
        return {"total_requests": len(metrics), "successful_requests": 0}

    # Overall stats
    tokens_per_sec = [m["tokens_per_second"] for m in successful if m.get("tokens_per_second")]
    load_times = [m["load_time_ms"] for m in successful if m.get("load_time_ms")]
    gen_times = [m["generation_time_ms"] for m in successful if m.get("generation_time_ms")]

    # By model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for m in successful:
        by_model[m["model"]].append(m)

    model_stats = {}
    for model, model_metrics in by_model.items():
        model_tokens = [m["tokens_per_second"] for m in model_metrics if m.get("tokens_per_second")]
        model_load = [m["load_time_ms"] for m in model_metrics if m.get("load_time_ms")]
        model_gen = [m["generation_time_ms"] for m in model_metrics if m.get("generation_time_ms")]

        model_stats[model] = {
            "request_count": len(model_metrics),
            "avg_tokens_per_second": sum(model_tokens) / len(model_tokens) if model_tokens else 0,
            "avg_load_time_ms": sum(model_load) / len(model_load) if model_load else 0,
            "avg_generation_time_ms": sum(model_gen) / len(model_gen) if model_gen else 0,
            "min_tokens_per_second": min(model_tokens) if model_tokens else 0,
            "max_tokens_per_second": max(model_tokens) if model_tokens else 0,
        }

    return {
        "total_requests": len(metrics),
        "successful_requests": len(successful),
        "avg_tokens_per_second": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
        "min_tokens_per_second": min(tokens_per_sec) if tokens_per_sec else 0,
        "max_tokens_per_second": max(tokens_per_sec) if tokens_per_sec else 0,
        "avg_load_time_ms": sum(load_times) / len(load_times) if load_times else 0,
        "avg_generation_time_ms": sum(gen_times) / len(gen_times) if gen_times else 0,
        "by_model": model_stats,
    }


def print_performance_report(stats: dict):
    """Print formatted performance report."""
    print("\n" + "=" * 60)
    print("  Ollama Performance Report")
    print("=" * 60)

    print("\nOverall Statistics:")
    print(f"  Total Requests:     {stats.get('total_requests', 0):,}")
    print(f"  Successful:         {stats.get('successful_requests', 0):,}")

    if stats.get("avg_tokens_per_second"):
        print("\nGeneration Performance:")
        print(f"  Avg Tokens/sec:     {stats['avg_tokens_per_second']:.2f}")
        print(f"  Min Tokens/sec:     {stats.get('min_tokens_per_second', 0):.2f}")
        print(f"  Max Tokens/sec:     {stats.get('max_tokens_per_second', 0):.2f}")

    if stats.get("avg_load_time_ms"):
        print("\nTiming Breakdown:")
        print(f"  Avg Load Time:      {stats['avg_load_time_ms']:.2f}ms")
        print(f"  Avg Generation:    {stats['avg_generation_time_ms']:.2f}ms")

    if stats.get("by_model"):
        print("\nPerformance by Model:")
        for model, model_stats in stats["by_model"].items():
            print(f"\n  {model}:")
            print(f"    Requests:         {model_stats['request_count']:,}")
            print(f"    Avg Tokens/sec:   {model_stats['avg_tokens_per_second']:.2f}")
            print(f"    Avg Load Time:    {model_stats['avg_load_time_ms']:.2f}ms")
            print(f"    Avg Generation:   {model_stats['avg_generation_time_ms']:.2f}ms")
            if model_stats.get("min_tokens_per_second"):
                print(
                    f"    Range:            {model_stats['min_tokens_per_second']:.2f} - {model_stats['max_tokens_per_second']:.2f} tokens/sec"
                )

    print("\n" + "=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate performance report from Ollama performance logs"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/performance.jsonl",
        help="Path to performance log file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name",
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Time window in minutes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Performance log file not found: {log_file}")
        print("Run some requests with performance tracking to generate logs.")
        sys.exit(1)

    # Parse logs
    metrics = parse_performance_log(log_file)

    # Filter by model if specified
    if args.model:
        metrics = [m for m in metrics if m.get("model") == args.model]

    # Filter by time window if specified
    if args.window:
        cutoff = datetime.now() - timedelta(minutes=args.window)
        metrics = [m for m in metrics if datetime.fromisoformat(m["timestamp"]) >= cutoff]

    # Calculate stats
    stats = calculate_performance_stats(metrics)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_performance_report(stats)


if __name__ == "__main__":
    main()
