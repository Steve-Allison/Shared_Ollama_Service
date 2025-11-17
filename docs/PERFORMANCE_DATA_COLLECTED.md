# Performance Data Collection

## Overview

The Shared Ollama Service collects comprehensive performance data through multiple telemetry systems. This document outlines what data is collected, where it's stored, and how to access it.

## Data Collection Systems

### 1. Structured Request Logging (`logs/requests.jsonl`)

**Location**: `logs/requests.jsonl`
**Format**: JSON Lines (one JSON object per line)
**Purpose**: Detailed request-level logging for all API operations

**Fields Collected** (per request):

#### Basic Request Info
- `event`: Event type (e.g., "api_request", "ollama_request")
- `timestamp`: ISO 8601 timestamp (UTC)
- `request_id`: Unique request identifier (UUID)
- `operation`: Operation type ("generate", "chat", "list_models")
- `status`: Request status ("success", "error")
- `client_type`: Client type ("rest_api", "async", "sync")
- `client_ip`: Client IP address
- `project_name`: Project name (from `X-Project-Name` header)

#### Performance Metrics
- `latency_ms`: Total request latency in milliseconds
- `model`: Model name used (e.g., "qwen2.5vl:7b")
- `model_load_ms`: Model load time in milliseconds (NEW - just added)
- `model_warm_start`: Boolean indicating if model was already loaded (NEW - just added)

#### Error Information (when status="error")
- `error_type`: Error type (e.g., "ValueError", "ConnectionError")
- `error_message`: Error message or description

**Example Entry**:
```json
{
  "event": "api_request",
  "timestamp": "2025-11-16T08:15:19.260699+00:00",
  "request_id": "b09c1dbc-64c0-4055-b692-ea30d7394df8",
  "operation": "chat",
  "status": "success",
  "client_type": "rest_api",
  "client_ip": "127.0.0.1",
  "project_name": "Docling_Machine",
  "model": "qwen2.5vl:7b",
  "latency_ms": 11084.258,
  "model_load_ms": 4242.467,
  "model_warm_start": false
}
```

### 2. In-Memory Metrics Collection (`MetricsCollector`)

**Location**: In-memory (class-level storage)
**Purpose**: Real-time metrics aggregation and statistics
**Storage**: Max 10,000 metrics (oldest trimmed automatically)

**Data Collected**:
- `model`: Model name
- `operation`: Operation type
- `latency_ms`: Request latency
- `success`: Boolean success flag
- `error`: Error type (if failed)
- `timestamp`: Request timestamp (UTC)

**Available Statistics**:
- Total requests count
- Successful/failed requests
- Requests by model
- Requests by operation
- Average latency (mean)
- Percentiles: P50 (median), P95, P99
- Errors by type
- First/last request timestamps

**Access**: Via `MetricsCollector.get_metrics()` or `MetricsCollector.get_service_metrics()`

### 3. Project Analytics (`AnalyticsCollector`)

**Location**: In-memory with project metadata
**Purpose**: Project-level usage tracking and analytics

**Data Collected**:
- All metrics from `MetricsCollector`
- `project`: Project name association
- Project-specific aggregations

**Available Reports**:
- `AnalyticsReport` with:
  - Total requests by project
  - Requests by model per project
  - Requests by operation per project
  - Latency statistics per project
  - Error rates per project

**Access**: Via `AnalyticsCollector.get_analytics(window_minutes, project)`

### 4. Detailed Performance Metrics (`PerformanceCollector`)

**Location**: `logs/performance.jsonl` (if used)
**Purpose**: Detailed timing breakdowns for performance analysis

**Data Collected** (when available):
- `model`: Model name
- `operation`: Operation type
- `timestamp`: Request timestamp
- `total_latency_ms`: Total request latency
- `load_duration_ns`: Model load duration (nanoseconds)
- `prompt_eval_count`: Number of prompt tokens
- `prompt_eval_duration_ns`: Prompt evaluation time
- `eval_count`: Number of generated tokens
- `eval_duration_ns`: Generation time
- `total_duration_ns`: Total generation duration
- `tokens_per_second`: Calculated throughput
- `success`: Boolean success flag
- `error`: Error message (if failed)

**Note**: Currently not actively used in use cases, but infrastructure exists.

### 5. Queue Statistics (`RequestQueue`)

**Location**: In-memory
**Purpose**: Request queue performance monitoring

**Data Collected**:
- `queued`: Currently queued requests
- `in_progress`: Currently processing requests
- `completed`: Total completed requests
- `failed`: Total failed requests
- `rejected`: Total rejected requests (queue full)
- `timeout`: Total timed-out requests
- `total_wait_time_ms`: Cumulative wait time
- `max_wait_time_ms`: Maximum wait time observed
- `avg_wait_time_ms`: Average wait time
- `max_concurrent`: Configuration (currently 6)
- `max_queue_size`: Configuration (currently 50)
- `default_timeout`: Configuration (currently 60.0 seconds)

**Access**: Via `/api/v1/queue/stats` endpoint

## Data Flow

```
Request → Use Case → Client → Ollama
    ↓
┌─────────────────────────────────────┐
│  Structured Logging                 │
│  (logs/requests.jsonl)              │
│  - Full request details             │
│  - Performance metrics              │
│  - Error information                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Metrics Collector                  │
│  (In-memory)                        │
│  - Aggregated statistics            │
│  - Percentiles                      │
│  - Error tracking                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Analytics Collector                │
│  (In-memory + project metadata)     │
│  - Project-level analytics          │
│  - Usage tracking                   │
└─────────────────────────────────────┘
```

## Recently Added Metrics

As of the latest update, the following metrics were added to structured logging:

1. **`model_load_ms`**: Model load time in milliseconds
   - Extracted from Ollama's `load_duration` response field
   - Converted from nanoseconds to milliseconds
   - `None` if not available

2. **`model_warm_start`**: Boolean indicating warm start
   - `True` if `load_duration == 0` (model already loaded)
   - `False` if model was loaded during request
   - `None` if load duration not available

These metrics enable tracking:
- Warm start rate (percentage of requests with warm models)
- Model loading overhead
- Performance impact of cold starts

## Accessing Performance Data

### Via Log Files

```bash
# View recent requests
tail -f logs/requests.jsonl | jq

# Analyze warm start rate
cat logs/requests.jsonl | jq -r 'select(.model_warm_start == true) | .model_warm_start' | wc -l

# Calculate average latency
cat logs/requests.jsonl | jq -r '.latency_ms' | awk '{sum+=$1; count++} END {print sum/count}'

# Filter by project
cat logs/requests.jsonl | jq 'select(.project_name == "Docling_Machine")'
```

### Via API Endpoints

```bash
# Queue statistics
curl http://localhost:8000/api/v1/queue/stats | jq

# Health check (includes service status)
curl http://localhost:8000/api/v1/health | jq
```

### Via Python Code

```python
from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.analytics import AnalyticsCollector

# Get service metrics
metrics = MetricsCollector.get_service_metrics(window_minutes=60)
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.average_latency_ms}ms")
print(f"P95 latency: {metrics.p95_latency_ms}ms")

# Get project analytics
analytics = AnalyticsCollector.get_analytics(
    window_minutes=60,
    project="Docling_Machine"
)
print(f"Project requests: {analytics.total_requests}")
```

## Performance Metrics Summary

### Currently Collected

✅ **Request-Level Metrics**:
- Latency (total request time)
- Model load time (NEW)
- Warm start indicator (NEW)
- Operation type
- Model name
- Success/failure status
- Error information

✅ **Aggregated Metrics**:
- Request counts (total, by model, by operation)
- Latency statistics (mean, median, P95, P99)
- Error rates and types
- Queue statistics

✅ **Project-Level Metrics**:
- Usage by project
- Performance by project
- Error rates by project

### Not Currently Collected (Available Infrastructure)

⚠️ **Detailed Performance Metrics**:
- Token-level metrics (tokens/second)
- Prompt evaluation time breakdown
- Generation time breakdown
- Detailed timing breakdowns

**Note**: Infrastructure exists (`PerformanceCollector`) but is not actively used in use cases.

## Data Retention

- **Structured Logs** (`requests.jsonl`): Persistent, grows unbounded (no rotation)
- **In-Memory Metrics**: Max 10,000 entries (oldest trimmed)
- **Queue Stats**: Real-time only (not persisted)

## Recommendations

1. **Monitor `logs/requests.jsonl`** for detailed request analysis
2. **Use queue stats endpoint** for real-time queue health
3. **Query metrics collector** for aggregated statistics
4. **Consider log rotation** for `requests.jsonl` in production
5. **Enable `PerformanceCollector`** if detailed token-level metrics are needed

