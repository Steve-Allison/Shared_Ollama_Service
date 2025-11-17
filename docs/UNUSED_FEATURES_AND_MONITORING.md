# Unused Features & Monitoring Capabilities

## Overview

The Shared Ollama Service has several powerful monitoring, analytics, and export features that are **fully implemented but not currently being used**. This document identifies these features and explains how to enable them.

## 🔴 Not Currently Used

### 1. AnalyticsCollector - Project-Based Analytics

**Status**: ❌ **Not Used** - Infrastructure exists but not integrated

**What It Does**:
- Tracks requests by project (via `X-Project-Name` header)
- Provides project-level analytics and breakdowns
- Generates hourly time-series metrics
- Supports filtering by time window and project

**Available Features**:
- `AnalyticsCollector.record_request_with_project()` - Record metrics with project association
- `AnalyticsCollector.get_analytics()` - Get comprehensive analytics report
- `AnalyticsCollector.export_json()` - Export analytics to JSON file
- `AnalyticsCollector.export_csv()` - Export analytics to CSV file
- `get_analytics_json()` - Convenience function for JSON export

**What We're Missing**:
- Project tracking is not enabled (we log `project_name` but don't use `AnalyticsCollector`)
- No API endpoint to query analytics
- No automatic project-based aggregation
- No hourly time-series data collection

**How to Enable**:
```python
# In use_cases.py, replace MetricsCollector with AnalyticsCollector
from shared_ollama.telemetry.analytics import AnalyticsCollector

# Instead of:
self._metrics.record_request(...)

# Use:
AnalyticsCollector.record_request_with_project(
    model=model_used,
    operation="chat",
    latency_ms=latency_ms,
    success=True,
    project=project_name,  # ← This enables project tracking
)
```

**Benefits**:
- Project-level usage tracking
- Hourly time-series metrics
- Export capabilities (JSON/CSV)
- Filter by project and time window

---

### 2. Performance Stats API Endpoint

**Status**: ❌ **Not Exposed** - `PerformanceCollector.get_performance_stats()` exists but no API endpoint

**What It Does**:
- Aggregates detailed performance statistics
- Groups by model
- Calculates average tokens/second, load times, generation times
- Provides per-model breakdowns

**Available Function**:
- `PerformanceCollector.get_performance_stats()` - Returns dict with:
  - `avg_tokens_per_second`: Overall average throughput
  - `avg_load_time_ms`: Average model load time
  - `avg_generation_time_ms`: Average generation time
  - `total_requests`: Count of successful requests
  - `by_model`: Per-model statistics

**What We're Missing**:
- No `/api/v1/performance/stats` endpoint
- Can't query performance stats via API
- Stats only available via Python code

**How to Enable**:
```python
# Add to src/shared_ollama/api/server.py
from shared_ollama.telemetry.performance import get_performance_stats

@app.get("/api/v1/performance/stats", tags=["Performance"])
async def get_performance_stats_endpoint():
    """Get aggregated performance statistics."""
    return get_performance_stats()
```

**Benefits**:
- Query performance stats via API
- Monitor tokens/second, load times
- Track performance trends over time

---

### 3. Metrics API Endpoint

**Status**: ❌ **Not Exposed** - `get_metrics_endpoint()` exists but no route

**What It Does**:
- Returns comprehensive service metrics
- Includes latency percentiles (P50, P95, P99)
- Provides request counts by model/operation
- Supports time-window filtering

**Available Function**:
- `get_metrics_endpoint(window_minutes=None)` - Returns dict with:
  - Total requests, successful/failed counts
  - Average latency, percentiles
  - Requests by model and operation
  - Error breakdowns

**What We're Missing**:
- No `/api/v1/metrics` endpoint
- Can't query metrics via API
- Metrics only available via Python code

**How to Enable**:
```python
# Add to src/shared_ollama/api/server.py
from shared_ollama.telemetry.metrics import get_metrics_endpoint

@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics(
    window_minutes: int | None = None,
):
    """Get service metrics."""
    return get_metrics_endpoint(window_minutes)
```

**Benefits**:
- Query metrics via API
- Monitor service health
- Track latency percentiles
- Filter by time window

---

### 4. Analytics API Endpoints

**Status**: ❌ **Not Exposed** - Full analytics infrastructure exists but no endpoints

**What It Does**:
- Project-based analytics
- Hourly time-series data
- Comprehensive reports with percentiles
- Export capabilities

**Available Functions**:
- `AnalyticsCollector.get_analytics(window_minutes, project)` - Full analytics report
- `get_analytics_json(window_minutes, project)` - JSON format
- `AnalyticsCollector.export_json(filepath, ...)` - Export to file
- `AnalyticsCollector.export_csv(filepath, ...)` - Export to CSV

**What We're Missing**:
- No `/api/v1/analytics` endpoint
- No `/api/v1/analytics/export` endpoint
- Can't query analytics via API
- No automatic exports

**How to Enable**:
```python
# Add to src/shared_ollama/api/server.py
from shared_ollama.telemetry.analytics import get_analytics_json

@app.get("/api/v1/analytics", tags=["Analytics"])
async def get_analytics(
    window_minutes: int | None = None,
    project: str | None = None,
):
    """Get analytics report."""
    return get_analytics_json(window_minutes, project)

@app.get("/api/v1/analytics/export/json", tags=["Analytics"])
async def export_analytics_json(
    window_minutes: int | None = None,
    project: str | None = None,
):
    """Export analytics to JSON file."""
    filepath = f"logs/analytics_export_{datetime.now().isoformat()}.json"
    path = AnalyticsCollector.export_json(filepath, window_minutes, project)
    return {"filepath": str(path), "message": "Analytics exported"}
```

**Benefits**:
- Query analytics via API
- Export analytics data
- Project-level insights
- Time-series analysis

---

### 5. Context Manager Tracking Functions

**Status**: ⚠️ **Available but Not Used** - Convenience functions exist

**Available Functions**:
- `track_performance(model, operation)` - Context manager for performance tracking
- `track_request(model, operation)` - Context manager for request tracking
- `track_request_with_project(model, operation, project)` - Context manager with project

**What We're Missing**:
- Not used in use cases (we call collectors directly)
- Could simplify code with context managers

**Example Usage**:
```python
# Instead of manual timing:
start_time = time.perf_counter()
result = await self._client.chat(...)
latency_ms = (time.perf_counter() - start_time) * 1000
PerformanceCollector.record_performance(...)

# Could use:
with track_performance(model, "chat"):
    result = await self._client.chat(...)
# Automatically records performance
```

---

## 📊 Current vs Available

### Currently Used ✅
- `MetricsCollector.record_request()` - Basic metrics
- `PerformanceCollector.record_performance()` - Detailed performance (just enabled)
- Structured logging (`log_request_event()`) - Request logs
- Queue stats endpoint (`/api/v1/queue/stats`)

### Available but Not Used ❌
- `AnalyticsCollector` - Project-based analytics
- `get_metrics_endpoint()` - Metrics API endpoint
- `get_performance_stats()` - Performance stats API endpoint
- `get_analytics_json()` - Analytics API endpoint
- Export functions (JSON/CSV)
- Context manager tracking functions

---

## 🎯 Recommended Next Steps

### Priority 1: Enable AnalyticsCollector
**Impact**: High - Enables project tracking and analytics
**Effort**: Low - Just change one line in use cases

### Priority 2: Add Performance Stats Endpoint
**Impact**: Medium - Enables performance monitoring via API
**Effort**: Low - Add one endpoint

### Priority 3: Add Metrics Endpoint
**Impact**: Medium - Enables metrics querying via API
**Effort**: Low - Add one endpoint

### Priority 4: Add Analytics Endpoints
**Impact**: High - Enables full analytics via API
**Effort**: Medium - Add endpoints + enable AnalyticsCollector

### Priority 5: Use Context Managers
**Impact**: Low - Code simplification
**Effort**: Low - Refactor use cases

---

## Summary

**Total Unused Features**: 5 major feature sets
- AnalyticsCollector (project tracking)
- Performance stats API
- Metrics API
- Analytics API
- Context manager helpers

**Estimated Value**:
- Better observability
- Project-level insights
- API-based monitoring
- Export capabilities
- Time-series analysis

**Implementation Effort**: Low to Medium (mostly adding API endpoints)

