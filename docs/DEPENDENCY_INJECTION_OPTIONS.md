# FastAPI Dependency Injection Testing Options

## Problem Identified

After investigation, the issue is **specifically with `Annotated[Type, Depends(...)]` syntax when using `httpx.ASGITransport`**.

Test results show:
- ✅ `TestClient` works with both sync and async endpoints
- ✅ `AsyncClient` with `ASGITransport` works with old-style `dep: Type = Depends(...)`
- ❌ `AsyncClient` with `ASGITransport` **FAILS** with `Annotated[Type, Depends(...)]`
- ✅ Fresh app instances work correctly

## Viable Options

### Option 1: Use TestClient (Recommended for Most Cases) ⭐

**Pros:**
- Works perfectly with dependency overrides
- Supports both sync and async endpoints
- Simple and well-tested
- No special setup needed

**Cons:**
- Runs in a separate thread (not truly async)
- May not catch all async-specific issues
- Limited for testing true async behavior

**Implementation:**
```python
@pytest.fixture
def api_client(mock_async_client, mock_use_cases):
    """Use TestClient - works with all dependency patterns."""
    setup_dependency_overrides(...)
    with TestClient(app) as client:
        yield client
    cleanup_dependency_overrides(app)
```

**Status:** ✅ **RECOMMENDED** - This is the simplest and most reliable solution.

---

### Option 2: Change Dependency Syntax to Old Style

**Pros:**
- Works with AsyncClient + ASGITransport
- Maintains async testing benefits
- No changes to test infrastructure

**Cons:**
- Requires changing endpoint signatures
- Loses type annotation benefits of `Annotated`
- Less modern Python 3.13+ style

**Implementation:**
Change from:
```python
async def generate(
    use_case: Annotated[GenerateUseCase, Depends(get_generate_use_case)],
    queue: Annotated[RequestQueue, Depends(get_queue)],
):
```

To:
```python
async def generate(
    use_case: GenerateUseCase = Depends(get_generate_use_case),
    queue: RequestQueue = Depends(get_queue),
):
```

**Status:** ⚠️ **WORKABLE** - Requires code changes but solves the issue.

---

### Option 3: Use TestClient for Non-Streaming, AsyncClient Only for Streaming

**Pros:**
- Best of both worlds
- TestClient for most tests (reliable)
- AsyncClient only when needed (streaming)

**Cons:**
- Two different test patterns
- More complex fixture setup

**Implementation:**
```python
# For regular endpoints
@pytest.fixture
def api_client(...):
    with TestClient(app) as client:
        yield client

# For streaming endpoints only
@pytest_asyncio.fixture
async def async_api_client(...):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport) as client:
        yield client
```

**Status:** ✅ **RECOMMENDED** - Hybrid approach, most practical.

---

### Option 4: Create Test-Specific App Instance

**Pros:**
- Isolates test state
- Can customize app per test
- Works with all dependency patterns

**Cons:**
- More setup overhead
- Need to duplicate app creation
- May miss integration issues

**Implementation:**
```python
@pytest.fixture
def test_app():
    """Create a fresh app instance for testing."""
    from shared_ollama.api.server import create_app
    return create_app()

@pytest.fixture
def api_client(test_app):
    setup_dependency_overrides(test_app, ...)
    with TestClient(test_app) as client:
        yield client
    cleanup_dependency_overrides(test_app)
```

**Status:** ⚠️ **WORKABLE** - Requires refactoring app creation.

---

### Option 5: Use Dependency Injection at Route Registration

**Pros:**
- More explicit dependency management
- Works with all clients
- Better for complex scenarios

**Cons:**
- Significant refactoring required
- Changes architecture
- More complex setup

**Implementation:**
```python
# Register routes with dependencies explicitly
app.post("/api/v1/generate")(
    generate_endpoint_factory(use_case, queue)
)
```

**Status:** ❌ **NOT RECOMMENDED** - Too much refactoring.

---

## Recommended Solution: Option 1 + Option 3 Hybrid

### Strategy

1. **Use TestClient for all non-streaming endpoints** (Option 1)
   - Simple, reliable, works with all dependency patterns
   - Covers 95% of test cases

2. **Use AsyncClient only for streaming tests** (Option 3)
   - Only when truly needed for SSE streaming
   - Accept that streaming tests may need workarounds

3. **Keep current dependency syntax** (`Annotated[Type, Depends(...)]`)
   - Modern Python 3.13+ style
   - Better type hints
   - Works fine with TestClient

### Implementation Plan

1. Update all non-streaming tests to use `TestClient` (sync fixture)
2. Keep `AsyncClient` only for streaming endpoint tests
3. For streaming tests, either:
   - Accept TestClient limitations (may work for basic streaming)
   - Use workarounds for AsyncClient (e.g., test at adapter level)
   - Test streaming separately at integration level

### Code Changes Needed

1. **Revert async_api_client to sync TestClient for most tests**
2. **Create separate streaming_client fixture for streaming tests**
3. **Update test patterns accordingly**

---

## Quick Fix: Immediate Solution

The fastest fix is to **use TestClient for everything**:

```python
@pytest.fixture
def api_client(mock_async_client, mock_use_cases):
    """Use TestClient - works with all dependency patterns."""
    setup_dependency_overrides(...)
    with TestClient(app) as client:
        yield client
    cleanup_dependency_overrides(app)
```

This will make all tests pass immediately, and TestClient handles async endpoints correctly.

---

## Long-term Solution

If you need true async testing:
1. Wait for FastAPI fix for `Annotated[Type, Depends(...)]` with ASGITransport
2. Or change to old-style dependency syntax for endpoints that need async testing
3. Or use TestClient (which works fine for most cases)

---

## Decision Matrix

| Option | Complexity | Reliability | Maintainability | Recommendation |
|--------|-----------|------------|-----------------|----------------|
| Option 1: TestClient | Low | High | High | ⭐⭐⭐⭐⭐ |
| Option 2: Change Syntax | Medium | High | Medium | ⭐⭐⭐ |
| Option 3: Hybrid | Medium | High | High | ⭐⭐⭐⭐⭐ |
| Option 4: Test App | High | High | Medium | ⭐⭐⭐ |
| Option 5: Route DI | Very High | High | Low | ⭐ |

**Winner: Option 1 (TestClient) or Option 3 (Hybrid)**

