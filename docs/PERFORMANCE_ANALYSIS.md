# Performance Analysis & Optimization Opportunities

## Current Performance Metrics

Based on analysis of `logs/requests.jsonl`:

### Request Statistics
- **Total Requests**: 202
- **Operations**: 200 chat, 2 list_models
- **Primary Model**: qwen2.5vl:7b (198 requests)
- **Primary Project**: Docling_Machine (94 requests)
- **Success Rate**: 99% (200 success, 2 errors)

### Latency Analysis
- **Mean Latency**: 10,718ms (10.7 seconds)
- **Median Latency**: 11,036ms (11.0 seconds)
- **Min Latency**: 1.81ms
- **Max Latency**: 22,612ms (22.6 seconds)
- **Std Deviation**: 3,886ms

### Current Configuration (Already Optimized)
- ✅ **OLLAMA_KEEP_ALIVE**: Already set to **30m** in `ollama_manager.py` (line 166)
- ✅ **Warmup Script**: Available at `scripts/warmup_models.sh` (not auto-run)
- ⚠️ **Issue**: `model_load_ms` and `model_warm_start` not being logged to `requests.jsonl`

### Queue Performance
- **Average Wait Time**: 3.0 seconds
- **Max Wait Time**: 8.0 seconds
- **Max Concurrent**: 3 requests
- **Queue Size**: 50 requests
- **Rejections**: 0
- **Timeouts**: 0

## Performance Bottlenecks Identified

### 1. 🟡 MEDIUM: Missing Performance Metrics in Logs

**Problem:**
- `model_load_ms` and `model_warm_start` fields not being logged to `requests.jsonl`
- Cannot track actual warm start rate from logs
- Performance metrics exist in API responses but not in structured logs

**Impact:**
- Cannot analyze actual warm start performance
- Missing visibility into model loading times
- Hard to identify performance issues

**Solution:**
- Add `model_load_ms` and `model_warm_start` to structured logging in use cases
- These fields are already calculated in API responses, just need to log them

**Expected Improvement:**
- Better observability
- Can track actual warm start rate
- Identify performance bottlenecks

### 2. 🟡 MEDIUM: Low Concurrency Limit

**Problem:**
- `max_concurrent=3` limits parallel processing
- Average wait time: 3 seconds in queue
- May be too conservative if models are warm

**Current Configuration:**
```python
RequestQueue(max_concurrent=3, max_queue_size=50, default_timeout=60.0)
```

**Recommendation:**
- Monitor actual warm start rate first (after fixing logging)
- If warm start rate >80%, consider increasing to 5-6 concurrent requests
- Model loading is the bottleneck, not inference (when warm)

**Expected Improvement:**
- Reduced queue wait times
- Better throughput for burst traffic
- More efficient resource utilization

### 3. 🟢 LOW: Model Warmup Not Auto-Run

**Current State:**
- Warmup script exists: `scripts/warmup_models.sh`
- Not automatically called from `start.sh`
- Must be run manually

**Recommendation:**
- Consider adding warmup to service startup (optional)
- Or document that users should run it manually if needed
- First request will be cold, but subsequent requests benefit from 30m keep-alive

**Expected Improvement:**
- First request is warm (no load time)
- Consistent performance from start

### 4. 🟢 LOW: Request Pattern Optimization

**Observation:**
- Requests from Docling_Machine project (94 requests)
- Could benefit from connection pooling optimizations

**Recommendation:**
- Monitor request patterns
- Consider project-specific keep-alive settings
- Optimize for common use cases

## Recommended Optimizations

### Priority 1: Add Performance Metrics to Logging (HIGH)

**Action:**
Update `src/shared_ollama/application/use_cases.py` to log `model_load_ms` and `model_warm_start`:

```python
# In ChatUseCase.execute() and GenerateUseCase.execute()
self._logger.log_request({
    "event": "api_request",
    "client_type": "rest_api",
    "operation": "chat",
    "status": "success",
    "model": model_used,
    "request_id": request_id,
    "client_ip": client_ip,
    "project_name": project_name,
    "latency_ms": round(latency_ms, 3),
    "model_load_ms": result.get("load_duration", 0) / 1_000_000 if result.get("load_duration") else None,
    "model_warm_start": (result.get("load_duration", 0) == 0),
})
```

**Expected Impact:**
- Can track actual warm start rate
- Better observability into performance
- Identify bottlenecks

### Priority 2: Monitor and Tune Concurrency (After Priority 1)

**Action:**
1. After fixing logging, monitor warm start rate
2. If warm start rate >80%, consider increasing concurrency:
   ```python
   queue = RequestQueue(
       max_concurrent=6,  # Increase from 3
       max_queue_size=50,
       default_timeout=60.0
   )
   ```

**Expected Impact:**
- Reduced queue wait times (if warm start rate is high)
- Better handling of concurrent requests
- Improved throughput

### Priority 3: Optional - Auto-Run Model Warmup

**Action:**
Add to `scripts/start.sh` after server starts:
```bash
# Optional: Warm up models in background
if [ -f "$SCRIPT_DIR/warmup_models.sh" ]; then
    echo "Warming up models..."
    "$SCRIPT_DIR/warmup_models.sh" &
fi
```

**Expected Impact:**
- First request is warm
- Consistent performance from start

### Priority 4: Monitor and Tune

**Action:**
1. After fixing logging, monitor `logs/requests.jsonl` for warm start rate
2. Track latency improvements
3. `OLLAMA_KEEP_ALIVE` is already set to 30m (good default)
4. Adjust concurrency based on actual warm start rate

**Metrics to Track:**
- Warm start percentage (target: >80% after 30m keep-alive)
- Average latency (target: <3s for warm requests)
- Queue wait times (target: <1s average)
- Model load times (should be rare with 30m keep-alive)

## Performance Targets

### Current State
- Mean Latency: **10.7 seconds**
- Warm Start Rate: **Unknown** (not logged)
- Queue Wait: **3.0 seconds**
- OLLAMA_KEEP_ALIVE: **30m** (already configured)

### Target State (After Logging Fix)
- Mean Latency: **<3 seconds** (warm requests, <10s cold)
- Warm Start Rate: **>80%** (with 30m keep-alive)
- Queue Wait: **<1 second** (if concurrency increased)
- Model Load Time: **<10% of requests** (with 30m keep-alive)

## Usage Patterns

### Current Usage
- **Primary Model**: qwen2.5vl:7b (98% of requests)
- **Primary Project**: Docling_Machine (47% of requests)
- **Operation**: Chat (99% of requests)
- **Request Pattern**: Bursty (likely batch processing)

### Recommendations
1. **For Docling_Machine**: Consider longer keep-alive (60m) if batch processing
2. **For VLM Usage**: Optimize image preprocessing to reduce payload size
3. **For Chat**: Consider streaming for better UX on long responses

## Monitoring Commands

```bash
# Check current performance
python3 << 'EOF'
import json
warm = 0
total = 0
latencies = []
with open('logs/requests.jsonl') as f:
    for line in f:
        data = json.loads(line)
        total += 1
        if data.get('model_warm_start'):
            warm += 1
        if 'latency_ms' in data:
            latencies.append(data['latency_ms'])

print(f"Warm Start Rate: {warm/total*100:.1f}%")
if latencies:
    print(f"Mean Latency: {sum(latencies)/len(latencies)/1000:.2f}s")
EOF

# Check queue stats
curl http://localhost:8000/api/v1/queue/stats | python3 -m json.tool

# Monitor in real-time
tail -f logs/requests.jsonl | jq -r '[.operation, .model, .latency_ms, .model_warm_start] | @tsv'
```

## Next Steps

1. ✅ **Immediate**: Add `model_load_ms` and `model_warm_start` to structured logging
2. ✅ **After Priority 1**: Monitor actual warm start rate (should be >80% with 30m keep-alive)
3. ✅ **If warm start rate is high**: Consider increasing `max_concurrent` to 6
4. ✅ **Optional**: Add warmup script to service startup
5. ✅ **Ongoing**: Monitor performance metrics and tune based on actual usage patterns

## Key Findings

- ✅ **OLLAMA_KEEP_ALIVE is already optimized** (30m in `ollama_manager.py`)
- ✅ **Warmup script exists** but not auto-run (optional enhancement)
- ⚠️ **Performance metrics not logged** - need to add to structured logging
- ⚠️ **Cannot verify warm start rate** without logging fix

