# Performance Monitoring & Logging

This document describes what performance data we're currently collecting and what we can tell about Ollama performance.

## 📊 Current Performance Tracking

### What We're Tracking

#### 1. **Request-Level Metrics** (via `monitoring.py`)

**Captured**:
- ✅ Overall request latency (end-to-end)
- ✅ Success/failure status
- ✅ Model used
- ✅ Operation type (generate, chat, etc.)
- ✅ Error messages (if any)
- ✅ Timestamp

**Aggregated Statistics**:
- Total requests
- Success/failure rates
- Average latency (p50, p95, p99 percentiles)
- Requests by model
- Requests by operation
- Errors by type

**Limitation**: Only tracks **overall request time**, not Ollama's internal performance breakdown.

#### 2. **Ollama Internal Metrics** (via `GenerateResponse`)

**Available but NOT currently logged**:
- `total_duration` - Total request time (nanoseconds)
- `load_duration` - Model loading time (nanoseconds)
- `prompt_eval_count` - Tokens processed in prompt
- `prompt_eval_duration` - Prompt evaluation time (nanoseconds)
- `eval_count` - Tokens generated
- `eval_duration` - Generation time (nanoseconds)

**Status**: These metrics are **captured** in the response but **not stored or logged**.

### Current Logging

#### Ollama Service Logs (`logs/ollama.log`)

**What's logged**:
- HTTP request/response logs
- Request duration (end-to-end)
- Status codes
- Timestamps

**Example**:
```
[GIN] 2025/11/05 - 16:14:34 | 200 | 12.650958875s | ::1 | POST "/api/generate"
[GIN] 2025/11/05 - 16:14:46 | 200 |    10.512913s | ::1 | POST "/api/generate"
```

**Limitation**: Only shows overall request time, not internal breakdown.

#### Application Logging

**Python logging** (via `logging` module):
- Connection events
- Model generation start
- Errors and warnings
- Circuit breaker state changes

**Log level**: INFO (default)

## 🔍 What We Can Tell About Performance

### Currently Available

1. **Overall Request Performance**:
   - Average latency
   - P95/P99 latency
   - Success rates
   - Request patterns (by model, operation)

2. **Service Health**:
   - Service availability
   - Error rates
   - Connection issues

3. **Usage Patterns**:
   - Requests per model
   - Requests per operation
   - Time-series trends (hourly)

### What We CAN'T Tell (Missing)

1. **Model Loading Performance**:
   - How long model loading takes
   - Cold start vs warm start difference
   - Model switching overhead

2. **Generation Performance**:
   - Tokens per second
   - Prompt evaluation time vs generation time
   - Performance by prompt length

3. **Detailed Breakdown**:
   - Load time vs inference time
   - Prompt processing vs token generation
   - Memory/GPU utilization

## 🚀 Recommended Enhancements

### 1. Enhanced Performance Tracking

**Add Ollama internal metrics to monitoring**:

```python
@dataclass
class DetailedRequestMetrics(RequestMetrics):
    """Extended metrics with Ollama internal performance data."""
    
    # Ollama internal metrics
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    
    # Calculated metrics
    tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
```

### 2. Performance Logging

**Add structured logging for performance**:

```python
import json
import logging

# Structured performance logger
performance_logger = logging.getLogger("ollama.performance")
performance_logger.setLevel(logging.INFO)

# Log detailed metrics
performance_logger.info(json.dumps({
    "model": "qwen2.5vl:7b",
    "load_duration_ms": load_duration / 1_000_000,
    "prompt_eval_duration_ms": prompt_eval_duration / 1_000_000,
    "eval_duration_ms": eval_duration / 1_000_000,
    "tokens_per_second": tokens_per_second,
    "prompt_tokens": prompt_eval_count,
    "generated_tokens": eval_count,
}))
```

### 3. Performance Dashboard

**Enhanced analytics with performance breakdown**:

```python
# Add to analytics.py
@dataclass
class PerformanceMetrics:
    """Performance-specific metrics."""
    
    avg_load_time_ms: float = 0.0
    avg_prompt_eval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_prompt_tokens_per_second: float = 0.0
    
    # By model
    performance_by_model: dict[str, PerformanceMetrics] = field(default_factory=dict)
```

### 4. Log Aggregation

**Structured logging to file**:

```python
# Performance log file
performance_handler = logging.FileHandler("logs/performance.jsonl")
performance_handler.setFormatter(logging.Formatter("%(message)s"))

# Log as JSON lines for easy parsing
performance_logger.addHandler(performance_handler)
```

## 📈 Metrics We Should Track

### High Priority

1. **Model Loading Time** (`load_duration`)
   - Cold start performance
   - Warm start performance
   - Model switching overhead

2. **Token Generation Speed** (`tokens_per_second`)
   - Calculated from `eval_count` / `eval_duration`
   - Performance indicator
   - Model comparison

3. **Prompt Processing Speed** (`prompt_tokens_per_second`)
   - Calculated from `prompt_eval_count` / `prompt_eval_duration`
   - Understanding prompt complexity impact

### Medium Priority

4. **Performance by Model**
   - Compare qwen2.5vl:7b vs qwen2.5:14b
   - Identify performance differences

5. **Performance by Prompt Length**
   - Impact of prompt size on latency
   - Optimization opportunities

6. **Time-series Performance Trends**
   - Performance degradation over time
   - Peak usage impact

## 🛠️ Implementation Status

### ✅ Currently Implemented

- Basic request latency tracking
- Success/failure tracking
- Aggregate statistics (p50, p95, p99)
- Usage patterns (by model, operation)
- Ollama service logs (HTTP request logs)

### ⏳ Missing / Not Implemented

- Ollama internal metrics storage
- Detailed performance breakdown logging
- Token generation speed calculation
- Model loading time tracking
- Structured performance logs
- Performance dashboard with internal metrics

## 💡 Quick Wins

1. **Add Ollama metrics to monitoring** (1 hour)
   - Extend `RequestMetrics` to include Ollama internal metrics
   - Update `track_request` to capture these

2. **Performance logging** (30 minutes)
   - Add structured logging for performance metrics
   - Log to `logs/performance.jsonl`

3. **Enhanced analytics** (1 hour)
   - Add performance metrics to analytics dashboard
   - Calculate tokens/sec, load times, etc.

## 📝 Current Log Files

- `logs/ollama.log` - Ollama HTTP request logs (3,538 lines)
- `logs/ollama.error.log` - Ollama error logs (743 lines)
- `logs/performance.jsonl` - Structured performance logs (if using `performance_logging.py`)
- Python application logs (stdout/stderr)

## ✅ Enhanced Performance Tracking (New)

**File**: `performance_logging.py`

**Features**:
- Captures Ollama internal metrics (load_duration, eval_duration, tokens/sec)
- Structured JSON logging to `logs/performance.jsonl`
- Performance statistics aggregation
- Model-level performance comparison

**Usage**:
```python
from performance_logging import track_performance, get_performance_stats
from shared_ollama_client import SharedOllamaClient

client = SharedOllamaClient()
response = client.generate("Hello!")

# Track with detailed metrics
PerformanceCollector.record_performance(
    model="qwen2.5vl:7b",
    operation="generate",
    total_latency_ms=500.0,
    success=True,
    response=response,  # Includes Ollama internal metrics
)

# Get performance stats
stats = get_performance_stats()
print(f"Avg tokens/sec: {stats['avg_tokens_per_second']}")
```

**Performance Report**:
```bash
python scripts/performance_report.py
python scripts/performance_report.py --model qwen2.5vl:7b
```

