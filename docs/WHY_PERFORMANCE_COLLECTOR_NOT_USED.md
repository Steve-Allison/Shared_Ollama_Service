# Why PerformanceCollector Is Not Currently Used

## The Problem

`PerformanceCollector` exists and has full infrastructure, but it's **not being called** in the use cases. Here's why:

## Root Cause: Type Mismatch

### What PerformanceCollector Expects

`PerformanceCollector.record_performance()` requires a `GenerateResponse` object:

```python
def record_performance(
    cls,
    model: str,
    operation: str,
    total_latency_ms: float,
    success: bool,
    response: GenerateResponse | None = None,  # ← Expects GenerateResponse object
    error: str | None = None,
) -> None:
```

The `GenerateResponse` object contains:
- `load_duration`: int (nanoseconds)
- `prompt_eval_count`: int
- `prompt_eval_duration`: int (nanoseconds)
- `eval_count`: int
- `eval_duration`: int (nanoseconds)
- `total_duration`: int (nanoseconds)

### What Use Cases Actually Receive

The use cases receive **dicts**, not `GenerateResponse` objects:

1. **Adapter converts GenerateResponse → dict** (`infrastructure/adapters.py` line 114-122):
   ```python
   # Convert GenerateResponse to dict (preserving all fields)
   return {
       "text": result.text,
       "model": result.model,
       "prompt_eval_count": result.prompt_eval_count,
       "eval_count": result.eval_count,
       "total_duration": result.total_duration,
       "load_duration": result.load_duration,
   }
   ```

2. **Use cases receive dicts** from the adapter:
   ```python
   result = await self._client.generate(...)  # Returns dict, not GenerateResponse
   ```

3. **PerformanceCollector can't use dicts** - it needs the `GenerateResponse` object type.

## Current Architecture Flow

```
Use Case → Client Adapter → Async Client → Ollama API
    ↓
Adapter converts GenerateResponse → dict
    ↓
Use Case receives dict
    ↓
Use Case logs to MetricsCollector (works with dicts)
    ↓
PerformanceCollector (NOT CALLED - needs GenerateResponse object)
```

## Why This Design?

The adapter converts `GenerateResponse` to dict for:
1. **Consistency**: All adapters return dicts (not domain objects)
2. **Simplicity**: Use cases work with simple dicts
3. **Flexibility**: Dicts are easier to serialize/log

But this breaks `PerformanceCollector` which expects typed objects.

## Solutions

### Option 1: Modify PerformanceCollector to Accept Dicts (RECOMMENDED)

**Pros:**
- Minimal changes
- Works with current architecture
- No breaking changes

**Implementation:**
```python
@classmethod
def record_performance(
    cls,
    model: str,
    operation: str,
    total_latency_ms: float,
    success: bool,
    response: GenerateResponse | dict[str, Any] | None = None,  # ← Accept dicts too
    error: str | None = None,
) -> None:
    # Handle both GenerateResponse and dict
    if isinstance(response, dict):
        load_duration = response.get("load_duration")
        prompt_eval_count = response.get("prompt_eval_count")
        # ... extract other fields
    elif isinstance(response, GenerateResponse):
        load_duration = response.load_duration
        prompt_eval_count = response.prompt_eval_count
        # ... use object attributes
```

### Option 2: Keep GenerateResponse Objects in Adapter

**Pros:**
- Type safety
- PerformanceCollector works as-is

**Cons:**
- Breaks current architecture (adapters return dicts)
- Requires changes to use cases
- More complex

### Option 3: Convert Dict Back to GenerateResponse

**Pros:**
- PerformanceCollector works as-is

**Cons:**
- Inefficient (converting back and forth)
- Requires creating GenerateResponse from dict
- Adds complexity

## ✅ SOLUTION IMPLEMENTED

**Option 1 has been implemented**: Modified `PerformanceCollector.record_performance()` to accept both `GenerateResponse` objects and dicts.

### Changes Made

1. **Updated `PerformanceCollector.record_performance()`** (`src/shared_ollama/telemetry/performance.py`):
   - Changed signature to accept `response: GenerateResponse | dict[str, Any] | None`
   - Added logic to handle both dicts and GenerateResponse objects
   - Maintains backward compatibility

2. **Integrated into Use Cases** (`src/shared_ollama/application/use_cases.py`):
   - Added `PerformanceCollector.record_performance()` calls in `GenerateUseCase.execute()`
   - Added `PerformanceCollector.record_performance()` calls in `ChatUseCase.execute()`
   - Passes dict responses directly (adapter returns dicts)

### What We're Now Collecting

✅ **Token-level metrics**: Tokens per second (generation and prompt eval)
✅ **Timing breakdowns**: Prompt eval time vs generation time
✅ **Detailed performance logs**: `logs/performance.jsonl` now populated
✅ **Performance statistics**: Can query detailed performance stats
✅ **Model load times**: Detailed load duration tracking
✅ **Warm start tracking**: Via load_duration == 0

### Performance Data Now Available

- `load_duration_ns`: Model load time in nanoseconds
- `prompt_eval_count`: Number of prompt tokens
- `prompt_eval_duration_ns`: Prompt evaluation time
- `eval_count`: Number of generated tokens
- `eval_duration_ns`: Generation time
- `total_duration_ns`: Total generation duration
- `tokens_per_second`: Generation throughput
- `prompt_tokens_per_second`: Prompt evaluation throughput
- `load_time_ms`: Model load time (converted to ms)
- `prompt_eval_time_ms`: Prompt evaluation time (converted to ms)
- `generation_time_ms`: Generation time (converted to ms)

All metrics are logged to `logs/performance.jsonl` and available via `PerformanceCollector.get_performance_stats()`.

