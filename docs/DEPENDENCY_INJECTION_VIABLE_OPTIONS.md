# Viable Options for FastAPI Dependency Injection Testing

## Investigation Results

After thorough testing, I've identified the root cause and tested multiple solutions:

### Root Cause
The issue is **specifically with `Annotated[Type, Depends(...)]` syntax** when dependency overrides are used. This affects both:
- `httpx.ASGITransport` + `AsyncClient` ❌
- `TestClient` (in some cases) ⚠️

### Test Results Summary

| Approach | Simple Dependencies | Annotated Dependencies | Status |
|----------|-------------------|----------------------|--------|
| TestClient + `dep: Type = Depends(...)` | ✅ | ✅ | **WORKS** |
| TestClient + `Annotated[Type, Depends(...)]` | ✅ | ⚠️ | **WORKS (with caveats)** |
| AsyncClient + ASGITransport + old style | ✅ | ✅ | **WORKS** |
| AsyncClient + ASGITransport + Annotated | ❌ | ❌ | **FAILS** |

## Viable Options

### Option 1: Change Dependency Syntax (Recommended) ⭐⭐⭐⭐⭐

**Change from:**
```python
async def generate(
    use_case: Annotated[GenerateUseCase, Depends(get_generate_use_case)],
    queue: Annotated[RequestQueue, Depends(get_queue)],
):
```

**To:**
```python
async def generate(
    use_case: GenerateUseCase = Depends(get_generate_use_case),
    queue: RequestQueue = Depends(get_queue),
):
```

**Pros:**
- ✅ Works with both TestClient and AsyncClient
- ✅ Works with dependency overrides
- ✅ No test infrastructure changes needed
- ✅ Simple, reliable solution

**Cons:**
- ⚠️ Loses some type annotation benefits (but still type-hinted)
- ⚠️ Less "modern" Python 3.13+ style

**Implementation Effort:** Low (change ~3 endpoint signatures)

**Recommendation:** ⭐⭐⭐⭐⭐ **BEST OPTION** - Simple, reliable, works everywhere

---

### Option 2: Use TestClient for Everything ⭐⭐⭐⭐

**Keep current dependency syntax, use TestClient for all tests.**

**Pros:**
- ✅ No code changes to endpoints
- ✅ TestClient handles async endpoints correctly
- ✅ Works with dependency overrides (in most cases)
- ✅ Simple test infrastructure

**Cons:**
- ⚠️ May have issues with `Annotated[Type, Depends(...)]` in some edge cases
- ⚠️ TestClient runs in separate thread (not truly async)
- ⚠️ May need workarounds for complex scenarios

**Implementation Effort:** Medium (update all test fixtures)

**Recommendation:** ⭐⭐⭐⭐ **GOOD OPTION** - Works for most cases

---

### Option 3: Hybrid Approach ⭐⭐⭐

**Use TestClient for most tests, AsyncClient only for streaming.**

**Pros:**
- ✅ Best of both worlds
- ✅ TestClient for reliability
- ✅ AsyncClient for true async testing when needed

**Cons:**
- ⚠️ Two different test patterns
- ⚠️ More complex fixture setup
- ⚠️ Still need to solve AsyncClient issue for streaming

**Implementation Effort:** Medium-High

**Recommendation:** ⭐⭐⭐ **ACCEPTABLE** - More complex but flexible

---

### Option 4: Create Test-Specific App Instance ⭐⭐⭐

**Create a fresh app instance for each test with dependencies pre-configured.**

**Pros:**
- ✅ Isolated test state
- ✅ Can customize app per test
- ✅ Works with all dependency patterns

**Cons:**
- ⚠️ Requires refactoring app creation
- ⚠️ More setup overhead
- ⚠️ May miss integration issues

**Implementation Effort:** High (refactor app creation)

**Recommendation:** ⭐⭐⭐ **WORKABLE** - Requires significant refactoring

---

## Recommended Solution: Option 1

**Change dependency syntax from `Annotated[Type, Depends(...)]` to `Type = Depends(...)`**

### Why This Is Best

1. **Proven to work** - Tested and confirmed working
2. **Minimal changes** - Only 3 endpoint signatures need updating
3. **No test infrastructure changes** - All existing tests work
4. **Works with all clients** - TestClient, AsyncClient, ASGITransport
5. **Reliable** - No edge cases or workarounds needed

### Implementation

**Files to change:**
- `src/shared_ollama/api/server.py`:
  - `list_models` endpoint (line ~406) - already uses old style ✅
  - `generate` endpoint (line ~476-477) - change to old style
  - `chat` endpoint (line ~621-622) - change to old style

**Change:**
```python
# Before
async def generate(
    request: Request,
    use_case: Annotated[GenerateUseCase, Depends(get_generate_use_case)],
    queue: Annotated[RequestQueue, Depends(get_queue)],
) -> Response:

# After
async def generate(
    request: Request,
    use_case: GenerateUseCase = Depends(get_generate_use_case),
    queue: RequestQueue = Depends(get_queue),
) -> Response:
```

**Impact:**
- ✅ All tests will work immediately
- ✅ No test infrastructure changes needed
- ✅ Works with TestClient and AsyncClient
- ✅ Maintains full type hints (just different syntax)

---

## Alternative: Option 2 (If You Want to Keep Annotated Syntax)

If you prefer to keep the `Annotated[Type, Depends(...)]` syntax:

1. Use TestClient for all tests
2. Accept that some edge cases may need workarounds
3. Document the limitation
4. Consider reporting as a FastAPI issue

**This is less reliable but requires no endpoint changes.**

---

## Decision Matrix

| Option | Code Changes | Test Changes | Reliability | Effort | Recommendation |
|--------|-------------|--------------|-------------|--------|----------------|
| Option 1: Change Syntax | 3 endpoints | None | ⭐⭐⭐⭐⭐ | Low | ⭐⭐⭐⭐⭐ |
| Option 2: TestClient Only | None | All fixtures | ⭐⭐⭐⭐ | Medium | ⭐⭐⭐⭐ |
| Option 3: Hybrid | None | All fixtures | ⭐⭐⭐ | Medium-High | ⭐⭐⭐ |
| Option 4: Test App | App creation | All fixtures | ⭐⭐⭐⭐ | High | ⭐⭐⭐ |

---

## Next Steps

1. **Choose Option 1** (recommended) - Change 3 endpoint signatures
2. **Or choose Option 2** - Update all test fixtures to use TestClient
3. **Test the solution** - Run full test suite
4. **Document the decision** - Update testing docs

---

## Quick Reference

**Option 1 Implementation:**
```python
# Change these 3 endpoints in server.py:
# 1. generate endpoint (line ~476-477)
# 2. chat endpoint (line ~621-622)
# 3. list_models already done ✅

# From:
use_case: Annotated[Type, Depends(getter)]

# To:
use_case: Type = Depends(getter)
```

**Option 2 Implementation:**
- Keep current endpoint syntax
- Use TestClient for all tests (already done)
- May need workarounds for edge cases

