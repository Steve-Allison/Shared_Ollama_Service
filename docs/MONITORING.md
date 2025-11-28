# Monitoring Guide

This guide covers monitoring, metrics, and observability for the Shared Ollama Service.

## Quick Status Check

```bash
# Fast status overview (recommended)
./scripts/core/status.sh
```

Shows:
- Service health
- Available models
- Process information
- Memory usage
- Quick health test

## Logs

### View Logs

```bash
# View REST API logs
tail -f logs/api.log

# View error logs
tail -f logs/api.error.log

# View structured request log (JSON lines)
tail -f logs/requests.jsonl

# View performance logs
tail -f logs/performance.jsonl
```

### Log Locations

- **API request logs**: `logs/api.log`
- **Error logs**: `logs/api.error.log`
- **Performance logs**: `logs/performance.jsonl` (detailed performance metrics)
- **Structured request events**: `logs/requests.jsonl` (model timings, load durations)

## Metrics & Performance

### Request Metrics

**Request Metrics** (via monitoring):

- Overall latency (p50, p95, p99)
- Success/failure rates
- Usage by model and operation

### Performance Analysis

```bash
# View performance report
python scripts/performance_report.py

# Filter by model
python scripts/performance_report.py --model qwen3:14b-q4_K_M

# Last hour
python scripts/performance_report.py --window 60
```

### Quick Monitoring

- **Quick status**: `./scripts/core/status.sh` (fast overview)
- **Health checks**: `./scripts/diagnostics/health_check.sh` (comprehensive)
- **Model status**: `curl http://0.0.0.0:8000/api/v1/models`
- **Resource usage**: `top -pid $(pgrep ollama)` or Activity Monitor

## API Monitoring Endpoints

The service provides comprehensive monitoring endpoints for metrics, performance, and analytics:

### Service Metrics (`GET /api/v1/metrics`)

Get comprehensive service metrics including request counts, latency statistics, and error breakdowns:

```bash
# Get all metrics
curl http://0.0.0.0:8000/api/v1/metrics

# Get metrics from last hour
curl "http://0.0.0.0:8000/api/v1/metrics?window_minutes=60"
```

**Response includes**:

- Total requests, successful/failed counts
- Latency percentiles (P50, P95, P99)
- Requests by model and operation
- Error breakdowns by type
- First/last request timestamps

### Performance Statistics (`GET /api/v1/performance/stats`)

Get detailed performance metrics including token generation rates and timing breakdowns:

```bash
curl http://0.0.0.0:8000/api/v1/performance/stats
```

**Response includes**:

- Average tokens per second (generation throughput)
- Average model load time
- Average generation time
- Per-model breakdowns with request counts

### Analytics (`GET /api/v1/analytics`)

Get comprehensive analytics with project-level tracking and time-series data:

```bash
# Get all analytics
curl http://0.0.0.0:8000/api/v1/analytics

# Get analytics for specific project
curl "http://0.0.0.0:8000/api/v1/analytics?project=Docling_Machine"

# Get analytics from last hour
curl "http://0.0.0.0:8000/api/v1/analytics?window_minutes=60"

# Combined filters
curl "http://0.0.0.0:8000/api/v1/analytics?window_minutes=60&project=Docling_Machine"
```

**Response includes**:

- Total requests, success rates
- Latency percentiles (P50, P95, P99)
- Requests by model, operation, and project
- Project-level metrics (detailed breakdowns per project)
- Hourly time-series metrics
- Time range (start_time, end_time)

**Project Tracking**: Projects are identified via the `X-Project-Name` header in requests. Analytics automatically tracks usage by project.

### Queue Statistics (`GET /api/v1/queue/stats`)

Get real-time queue performance metrics:

```bash
curl http://0.0.0.0:8000/api/v1/queue/stats
```

**Response includes**:

- Current queue state (queued, in_progress)
- Historical counts (completed, failed, rejected, timeout)
- Wait time statistics (total, max, average)
- Configuration (max_concurrent, max_queue_size, timeout)

## Performance Data Collection

The service collects comprehensive performance data:

### Structured Logs (`logs/requests.jsonl`)

- Request-level metrics (latency, model, operation, status)
- Model load times (`model_load_ms`)
- Warm start indicators (`model_warm_start`)
- Project names (from `X-Project-Name` header)
- Client IP addresses
- Error information

### Performance Logs (`logs/performance.jsonl`)

- Detailed timing breakdowns (load, prompt eval, generation)
- Token-level metrics (tokens/second, prompt tokens/second)
- Per-model performance statistics
- Comprehensive performance data for analysis

### In-Memory Metrics

- Real-time aggregations (via `/api/v1/metrics`)
- Project-based analytics (via `/api/v1/analytics`)
- Performance statistics (via `/api/v1/performance/stats`)

## Python Monitoring

### Track Requests

```python
from shared_ollama import MetricsCollector, track_request

# Track a request
with track_request("qwen3-vl:8b-instruct-q4_K_M", "generate"):
    response = client.generate("Hello!")

# Get metrics
metrics = MetricsCollector.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
print(f"P95 latency: {metrics.p95_latency_ms:.2f}ms")
```

### Get Analytics

```python
from shared_ollama import AnalyticsCollector, get_analytics_json

# Get analytics report
analytics = get_analytics_json(window_minutes=60, project="Docling_Machine")
print(f"Total requests: {analytics['total_requests']}")
print(f"Success rate: {analytics['success_rate']:.2%}")
print(f"Average latency: {analytics['average_latency_ms']:.2f}ms")
print(f"P95 latency: {analytics['p95_latency_ms']:.2f}ms")

# Export analytics
AnalyticsCollector.export_json("analytics.json")
AnalyticsCollector.export_csv("analytics.csv")
```

## Project Tracking

Automatically enabled! Include `X-Project-Name` header in requests:

```python
import requests

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    headers={"X-Project-Name": "Docling_Machine"},
    json={"model": "qwen3:14b-q4_K_M", "messages": [...]}
)
```

**Features**:

- ✅ Project-level usage tracking (automatic with `X-Project-Name` header)
- ✅ Hourly time-series metrics
- ✅ Latency percentiles (P50, P95, P99)
- ✅ Success rates and error breakdowns
- ✅ JSON/CSV export capabilities
- ✅ Time-window filtering
- ✅ Per-project detailed metrics

## See Also

- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions
- [Resource Management](RESOURCE_MANAGEMENT.md) - Memory and performance tuning
- [Operations Guide](OPERATIONS.md) - Service operations and maintenance

