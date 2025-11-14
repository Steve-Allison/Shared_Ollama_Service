# Test Fixes Needed After Clean Architecture Refactoring

## Summary

After refactoring to Clean Architecture, many tests need to be updated to work with the new dependency injection system. The main issue is that FastAPI's `TestClient` has limitations with async dependencies.

## Main Issue

`TestClient` doesn't properly handle dependency injection when dependencies are declared as function parameters in async endpoints. The error shows:
```
'missing', 'loc': ('query', 'use_case'), 'msg': 'Field required'
```

This indicates FastAPI is treating dependencies as query parameters instead of injecting them.

## Solutions

### Option 1: Use httpx.AsyncClient (Recommended)

Replace `TestClient` with `httpx.AsyncClient` for async endpoints:

```python
@pytest.fixture
async def async_api_client(mock_async_client, mock_use_cases):
    """Create an async test client."""
    # ... setup dependencies ...
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
```

### Option 2: Restructure Dependencies

Make dependencies optional or use a different dependency injection pattern that works better with TestClient.

### Option 3: Use Dependency Overrides Correctly

Ensure all nested dependencies are overridden:

```python
app.dependency_overrides[get_client_adapter] = lambda: client_adapter
app.dependency_overrides[get_logger_adapter] = lambda: logger_adapter
app.dependency_overrides[get_metrics_adapter] = lambda: metrics_adapter
app.dependency_overrides[get_generate_use_case] = lambda: use_case
app.dependency_overrides[get_queue] = lambda: queue
```

## Test Status

- ✅ Health endpoint tests (no dependencies) - PASSING
- ✅ List models endpoint tests - MOSTLY PASSING
- ❌ Generate endpoint tests - FAILING (dependency injection issue)
- ❌ Chat endpoint tests - FAILING (dependency injection issue)
- ❌ Streaming tests - FAILING (dependency injection issue)

## Next Steps

1. Convert all tests to use `httpx.AsyncClient` instead of `TestClient`
2. Update test fixtures to properly set up async dependencies
3. Ensure all mocks work with the new adapter pattern
4. Update streaming tests to work with async clients

## Files to Update

- `tests/test_api_server.py` - Main API server tests
- `tests/test_api_streaming.py` - Streaming tests
- `tests/test_body_parsing.py` - Body parsing tests
- `tests/test_client_rest_integration.py` - Integration tests

