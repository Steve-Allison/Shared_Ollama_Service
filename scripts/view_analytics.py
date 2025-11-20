#!/usr/bin/env python3
"""
Analytics Dashboard CLI
======================

Interactive command-line dashboard for viewing analytics.

Usage:
    python scripts/view_analytics.py
    python scripts/view_analytics.py --project knowledge_machine
    python scripts/view_analytics.py --window 60 --export analytics.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).resolve().parents[1]
src_root = project_root / "src"
sys.path.insert(0, str(src_root))

from shared_ollama import AnalyticsCollector, get_analytics_json  # noqa: E402

MS_IN_SECOND = 1000.0


def format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable format."""
    if ms < 1:
        return f"{ms * MS_IN_SECOND:.0f}μs"
    if ms < MS_IN_SECOND:
        return f"{ms:.2f}ms"
    return f"{ms / MS_IN_SECOND:.2f}s"


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{title}")
    print("-" * 60)


def print_analytics_dashboard(
    window_minutes: int | None = None,
    project: str | None = None,
):
    """Print analytics dashboard."""
    analytics = AnalyticsCollector.get_analytics(window_minutes, project)

    print_header("Shared Ollama Service - Analytics Dashboard")

    # Time range
    if analytics.start_time and analytics.end_time:
        print_section("Time Range")
        print(f"  Start: {analytics.start_time.isoformat()}")
        print(f"  End:   {analytics.end_time.isoformat()}")
        duration = (analytics.end_time - analytics.start_time).total_seconds() / 60
        print(f"  Duration: {duration:.1f} minutes")

    # Overall metrics
    print_section("Overall Metrics")
    print(f"  Total Requests:     {analytics.total_requests:,}")
    print(f"  Successful:        {analytics.successful_requests:,}")
    print(f"  Failed:            {analytics.failed_requests:,}")
    print(f"  Success Rate:      {analytics.success_rate:.2%}")

    # Latency metrics
    print_section("Latency Metrics")
    print(f"  Average:           {format_duration(analytics.average_latency_ms)}")
    print(f"  P50 (Median):      {format_duration(analytics.p50_latency_ms)}")
    print(f"  P95:               {format_duration(analytics.p95_latency_ms)}")
    print(f"  P99:               {format_duration(analytics.p99_latency_ms)}")

    # Requests by model
    if analytics.requests_by_model:
        print_section("Requests by Model")
        for model, count in sorted(
            analytics.requests_by_model.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = (
                count / analytics.total_requests * 100 if analytics.total_requests > 0 else 0
            )
            print(f"  {model:20s} {count:6,} ({percentage:5.1f}%)")

    # Requests by operation
    if analytics.requests_by_operation:
        print_section("Requests by Operation")
        for op, count in sorted(
            analytics.requests_by_operation.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = (
                count / analytics.total_requests * 100 if analytics.total_requests > 0 else 0
            )
            print(f"  {op:20s} {count:6,} ({percentage:5.1f}%)")

    # Requests by project
    if analytics.requests_by_project:
        print_section("Requests by Project")
        for proj, count in sorted(
            analytics.requests_by_project.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = (
                count / analytics.total_requests * 100 if analytics.total_requests > 0 else 0
            )
            print(f"  {proj:20s} {count:6,} ({percentage:5.1f}%)")

    # Project-level details
    if analytics.project_metrics:
        print_section("Project-Level Details")
        for proj_name, pm in sorted(analytics.project_metrics.items()):
            print(f"\n  Project: {proj_name}")
            print(f"    Total Requests: {pm.total_requests:,}")
            print(
                f"    Success Rate:   {pm.successful_requests / pm.total_requests * 100:.1f}%"
                if pm.total_requests > 0
                else "    Success Rate:   N/A"
            )
            print(f"    Avg Latency:     {format_duration(pm.average_latency_ms)}")
            if pm.requests_by_model:
                print(f"    Models Used:     {', '.join(pm.requests_by_model.keys())}")

    # Time-series summary
    if analytics.hourly_metrics:
        print_section("Hourly Summary (Last 24 hours)")
        for hour_metric in analytics.hourly_metrics[-24:]:
            print(
                f"  {hour_metric.timestamp.strftime('%Y-%m-%d %H:00')}: "
                f"{hour_metric.requests_count:,} requests, "
                f"avg {format_duration(hour_metric.average_latency_ms)}"
            )

    print("\n" + "=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="View analytics dashboard for Shared Ollama Service"
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Time window in minutes (default: all time)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Filter by project name",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export analytics to JSON file",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export analytics to CSV file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted dashboard",
    )

    args = parser.parse_args()

    if args.json:
        # Output as JSON
        data = get_analytics_json(args.window, args.project)
        print(json.dumps(data, indent=2))
    else:
        # Print formatted dashboard
        print_analytics_dashboard(args.window, args.project)

    # Export if requested
    if args.export:
        path = AnalyticsCollector.export_json(args.export, args.window, args.project)
        print(f"\n✓ Exported analytics to {path}")

    if args.export_csv:
        path = AnalyticsCollector.export_csv(args.export_csv, args.window, args.project)
        print(f"\n✓ Exported analytics to CSV: {path}")


if __name__ == "__main__":
    main()
