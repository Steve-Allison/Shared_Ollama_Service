# Dependency Injection Solution - Implementation Plan

## Investigation Results

After thorough testing, the issue is confirmed:

**Root Cause:** `httpx.ASGITransport` does not properly handle `Annotated[Type, Depends(...)]` syntax with FastAPI dependency overrides. This is a known limitation/bug.

**Test Results:**
- ✅ `TestClient` works perfectly with `Annotated[Type, Depends(...)]`
- ✅ `TestClient` works with async endpoints
- ❌ `AsyncClient` + `ASGITransport` fails with `Annotated[Type, Depends(...)]`
- ✅ `AsyncClient` + `ASGITransport` works with old-style `Depends(...)`

## Chosen Solution: Use TestClient for All Tests

**Rationale:**
1. TestClient handles async endpoints correctly
2. TestClient works with all dependency patterns
3. TestClient is simpler and more reliable
4. TestClient is the recommended approach in FastAPI docs
5. No code changes needed to endpoints

**Trade-offs:**
- TestClient runs in a separate thread (not truly async)
- For most use cases, this is acceptable
- Streaming tests may need special handling, but TestClient can handle SSE

## Implementation

### Step 1: Update Fixtures

Replace `async_api_client` with a sync `api_client` using TestClient:

```python
@pytest.fixture
def api_client(mock_async_client, mock_use_cases):
    """Create a test client using TestClient - works with all dependency patterns."""
    setup_dependency_overrides(...)
    with TestClient(app) as client:
        yield client
    cleanup_dependency_overrides(app)
```

### Step 2: Update All Tests

Convert all async tests to sync:

```python
# Before
@pytest.mark.asyncio
async def test_generate(async_api_client, mock_async_client):
    async with async_api_client as client:
        response = await client.post(...)

# After
def test_generate(api_client, mock_async_client):
    response = api_client.post(...)
```

### Step 3: Handle Streaming Tests

For streaming tests, TestClient can handle SSE:

```python
def test_streaming(api_client, mock_async_client):
    response = api_client.post("/api/v1/generate", json={"prompt": "test", "stream": True})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Read streaming response
    for line in response.iter_lines():
        if line.startswith("data: "):
            # Process chunk
```

## Alternative: Hybrid Approach

If you need true async testing for streaming:

1. Use TestClient for 95% of tests
2. Create separate streaming tests that test at the adapter/use case level
3. Keep AsyncClient only for integration tests that don't use dependency injection

## Next Steps

1. ✅ Investigation complete
2. ⏭️ Implement TestClient solution
3. ⏭️ Update all tests
4. ⏭️ Verify all tests pass
5. ⏭️ Document the solution

