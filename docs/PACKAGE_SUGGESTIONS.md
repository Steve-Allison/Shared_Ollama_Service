# Package Suggestions to Replace Custom Code

This document identifies custom code implementations that could be replaced with well-maintained, battle-tested packages.

## Summary

| Custom Implementation | Suggested Package | Lines Saved | Priority | Notes |
|----------------------|-------------------|-------------|----------|-------|
| LRU Cache (`OrderedDict`) | `cachetools` | ~100 | Medium | Better performance, thread-safe options |
| Exponential Backoff Retry | `tenacity` | ~70 | High | More features, better tested |
| Circuit Breaker | `pybreaker` or `circuitbreaker` | ~130 | High | Thread-safe, more robust |
| JSON Serialization | `orjson` | ~20 | Low | Performance boost, but stdlib works fine |
| Datetime Serialization | Already using Pydantic | 0 | N/A | Already optimized |

## Detailed Recommendations

### 1. LRU Cache → `cachetools`

**Current Implementation:**
- File: `src/shared_ollama/infrastructure/image_cache.py`
- Uses `collections.OrderedDict` manually
- ~126 lines of custom cache logic

**Suggested Package:**
```bash
pip install cachetools
```

**Benefits:**
- **Thread-safe**: Built-in thread-safe LRU cache (`TTLCache` with TTL support)
- **Better performance**: Optimized C implementation
- **Less code**: Reduces ~100 lines to ~30 lines
- **TTL support**: Native time-to-live support matches your needs

**Example Migration:**
```python
from cachetools import TTLCache

class ImageCache:
    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600.0):
        self._cache: TTLCache[str, CacheEntry] = TTLCache(
            maxsize=max_size,
            ttl=ttl_seconds
        )
        self._hits = 0
        self._misses = 0
    
    def get(self, data_url: str, target_format: str) -> tuple[str, ImageMetadata] | None:
        key = self._compute_key(data_url, target_format)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        return entry.base64_string, entry.metadata
    
    def put(self, data_url: str, target_format: str, base64_string: str, metadata: ImageMetadata) -> None:
        key = self._compute_key(data_url, target_format)
        self._cache[key] = CacheEntry(
            base64_string=base64_string,
            metadata=metadata,
            timestamp=time.time(),
        )
```

**Impact:** Reduces code complexity, improves thread safety, maintains same API.

---

### 2. Exponential Backoff Retry → `tenacity`

**Current Implementation:**
- File: `src/shared_ollama/core/resilience.py`
- Custom retry logic with exponential backoff and jitter
- ~70 lines

**Suggested Package:**
```bash
pip install tenacity
```

**Benefits:**
- **Battle-tested**: Used by major projects (requests, httpx)
- **More features**: Stop conditions, retry conditions, callbacks
- **Better logging**: Built-in retry logging
- **Async support**: Native async/await support
- **Less code**: Reduces ~70 lines to ~10-15 lines

**Example Migration:**
```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

logger = logging.getLogger(__name__)

# Replace exponential_backoff_retry function
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=60.0),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def exponential_backoff_retry(func: Callable[[], _T]) -> _T:
    """Execute function with exponential backoff retry logic."""
    return func()
```

**Impact:** More robust, better tested, easier to configure, supports async.

---

### 3. Circuit Breaker → `pybreaker` or `circuitbreaker`

**Current Implementation:**
- File: `src/shared_ollama/core/resilience.py`
- Custom circuit breaker with state machine
- ~130 lines

**Suggested Package:**
```bash
# Option 1: pybreaker (more features)
pip install pybreaker

# Option 2: circuitbreaker (simpler, more modern)
pip install circuitbreaker
```

**Benefits:**
- **Thread-safe**: Built-in thread safety
- **More features**: Callbacks, listeners, metrics
- **Better tested**: Used in production systems
- **Less code**: Reduces ~130 lines to ~20-30 lines

**Example Migration (using `circuitbreaker`):**
```python
from circuitbreaker import circuit

# Replace CircuitBreaker class usage
@circuit(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=ConnectionError
)
def resilient_operation():
    # Your operation here
    pass
```

**Or with `pybreaker` (more control):**
```python
from pybreaker import CircuitBreaker

# Create circuit breaker
breaker = CircuitBreaker(fail_max=5, timeout_duration=60.0)

# Use as decorator or context manager
@breaker
def resilient_operation():
    pass
```

**Impact:** Thread-safe, more robust, better tested, easier to maintain.

---

### 4. JSON Serialization → `orjson` (Optional)

**Current Implementation:**
- Multiple files using `json.dumps()` with custom defaults
- ~11 occurrences in source code

**Suggested Package:**
```bash
pip install orjson
```

**Benefits:**
- **Performance**: 2-3x faster than stdlib `json`
- **Better datetime handling**: Native datetime serialization
- **Type safety**: Better type handling

**Trade-offs:**
- **Dependency**: Adds another dependency
- **Compatibility**: Slight API differences
- **Current solution works**: Pydantic already handles this well

**Example Migration:**
```python
import orjson

# Replace json.dumps(event, default=_json_default)
orjson.dumps(event).decode('utf-8')

# Pydantic models already serialize well, so this is less critical
```

**Impact:** Performance boost, but current solution is adequate. Low priority.

---

## Implementation Priority

### High Priority (Immediate Benefits)

1. **`tenacity` for retry logic** ⭐⭐⭐
   - High impact, low risk
   - Better tested, more features
   - Reduces code complexity

2. **`pybreaker` or `circuitbreaker` for circuit breaker** ⭐⭐⭐
   - High impact, low risk
   - Thread-safe, production-ready
   - Significant code reduction

### Medium Priority (Nice to Have)

3. **`cachetools` for LRU cache** ⭐⭐
   - Medium impact, low risk
   - Better performance, thread-safe
   - Reduces code complexity

### Low Priority (Optional)

4. **`orjson` for JSON** ⭐
   - Low impact, adds dependency
   - Performance boost but current solution works
   - Can be deferred

---

## Migration Plan

### Phase 1: High Priority (Week 1)
1. Replace exponential backoff retry with `tenacity`
2. Replace circuit breaker with `pybreaker` or `circuitbreaker`
3. Update tests to verify behavior matches

### Phase 2: Medium Priority (Week 2)
1. Replace LRU cache with `cachetools`
2. Verify thread safety improvements
3. Update tests

### Phase 3: Low Priority (Future)
1. Consider `orjson` if JSON performance becomes bottleneck
2. Benchmark before/after to justify dependency

---

## Package Details

### `tenacity`
- **License**: Apache 2.0
- **Stars**: ~4.5k
- **Maintenance**: Active
- **Python**: 3.7+
- **Size**: Small (~50KB)

### `pybreaker`
- **License**: BSD
- **Stars**: ~1.2k
- **Maintenance**: Active
- **Python**: 3.7+
- **Size**: Small (~30KB)

### `circuitbreaker`
- **License**: MIT
- **Stars**: ~500
- **Maintenance**: Active
- **Python**: 3.8+
- **Size**: Very small (~10KB)

### `cachetools`
- **License**: MIT
- **Stars**: ~1.5k
- **Maintenance**: Active
- **Python**: 3.7+
- **Size**: Small (~40KB)

### `orjson`
- **License**: Apache 2.0 / MIT
- **Stars**: ~6k
- **Maintenance**: Very active
- **Python**: 3.8+
- **Size**: Medium (Rust-based, ~500KB)

---

## Notes

- All suggested packages are well-maintained and widely used
- All are compatible with Python 3.13 (your target version)
- Migration can be done incrementally without breaking changes
- Tests should be updated to verify equivalent behavior
- Consider adding these to `requirements.txt` or `pyproject.toml` optional dependencies

