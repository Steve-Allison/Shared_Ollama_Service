# Additional Opportunities Found

## Summary

After re-running the investigation, one additional opportunity was found:

| Custom Implementation | Suggested Replacement | Lines Saved | Priority | Notes |
|----------------------|---------------------|-------------|----------|-------|
| Custom percentile function | `statistics.quantiles()` | ~18 | Low | Script-only, not critical |

## Detailed Finding

### 1. Custom Percentile Function → `statistics.quantiles()`

**Current Implementation:**
- File: `scripts/async_load_test.py`
- Custom percentile calculation function (~18 lines)
- Uses manual interpolation algorithm

**Suggested Replacement:**
Use Python's standard library `statistics.quantiles()` function which is already used elsewhere in the codebase.

**Current Code:**
```python
def percentile(values: list[float], pct: float) -> float:
    """Return the percentile (0-1) value for a list of floats."""
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 1:
        return max(values)

    data = sorted(values)
    k = (len(data) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[f]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1
```

**Suggested Replacement:**
```python
import statistics

def percentile(values: list[float], pct: float) -> float:
    """Return the percentile (0-1) value for a list of floats."""
    if not values:
        return 0.0
    
    # Convert percentile (0-1) to quantile index
    # statistics.quantiles() returns list of quantiles for n divisions
    # For single percentile, use n=100 and index appropriately
    quantiles = statistics.quantiles(values, n=100, method='inclusive')
    index = int(pct * 100) - 1  # Convert 0-1 to 0-99 index
    index = max(0, min(index, len(quantiles) - 1))  # Clamp to valid range
    return quantiles[index]
```

**Or simpler approach:**
```python
import statistics

def percentile(values: list[float], pct: float) -> float:
    """Return the percentile (0-1) value for a list of floats."""
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
```

**Benefits:**
- Uses standard library (already imported elsewhere)
- Consistent with rest of codebase (metrics.py uses `statistics.quantiles()`)
- Less code to maintain
- Better tested (standard library implementation)

**Impact:** Low priority - script-only code, not used in production paths.

---

## Already Optimized ✅

The following areas are already using standard library or best practices:

1. **UUID Generation** - Using `uuid.uuid4()` ✅
2. **Hash Functions** - Using `hashlib.sha256()` ✅
3. **Statistics** - Using `statistics.quantiles()` in production code ✅
4. **Path Operations** - Using `pathlib.Path` throughout ✅
5. **Context Managers** - Using `@contextmanager` and `@asynccontextmanager` ✅
6. **Async Primitives** - Using `asyncio.Semaphore`, `asyncio.Queue` ✅
7. **Datetime** - Using `datetime.now(UTC)` ✅
8. **JSON Serialization** - Using Pydantic's built-in serialization ✅
9. **Environment Variables** - Using `pydantic-settings` ✅

---

## Conclusion

The codebase is already well-optimized. The only remaining opportunity is a minor one in a test script that could use standard library functions for consistency, but it's not critical since:

1. It's script-only code (not production)
2. The custom implementation works correctly
3. The benefit is primarily consistency, not functionality

All major custom implementations have been successfully replaced with battle-tested libraries:
- ✅ Retry logic → `tenacity`
- ✅ Circuit breaker → `circuitbreaker`
- ✅ LRU cache → `cachetools`
- ✅ Datetime serialization → Pydantic `TypeAdapter`
- ✅ Environment variables → `pydantic-settings`

