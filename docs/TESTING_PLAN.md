# Comprehensive Testing Plan

## Overview

This document outlines the testing strategy, reusable components, and patterns for the Shared Ollama Service test suite after the Clean Architecture refactoring.

## Testing Principles

1. **Behavior-Driven**: Tests verify behavior, not implementation details
2. **Isolation**: Each test is independent and can run in any order
3. **Reusability**: Common patterns extracted into reusable fixtures and utilities
4. **Async-First**: Use `httpx.AsyncClient` for async endpoints with dependencies
5. **Minimal Mocking**: Only mock external services (Ollama), not internal components

## Test Organization

### Test Categories

1. **Unit Tests** (`test_*.py`):
   - Domain entities and value objects
   - Use cases (with mocked dependencies)
   - Utility functions
   - Core components (queue, resilience)

2. **Integration Tests** (`test_*_integration.py`):
   - API endpoints with real use cases
   - Client-server interactions
   - End-to-end workflows

3. **Component Tests**:
   - Infrastructure adapters
   - Mappers
   - Dependency injection

## Reusable Components

### 1. Fixtures (in `conftest.py`)

#### Core Fixtures
- `mock_async_client`: Mock AsyncSharedOllamaClient
- `mock_use_cases`: Pre-configured use cases with mocked client
- `test_dependencies`: Complete dependency setup (adapters, queue)
- `async_api_client`: AsyncClient with dependency overrides
- `sync_api_client`: TestClient for sync endpoints (health, root)

#### Test Server Fixtures
- `ollama_server`: Mock Ollama HTTP server
- `ollama_config`: Test configuration
- `sample_models_response`: Sample API responses
- `sample_generate_response`: Sample generation responses
- `sample_chat_response`: Sample chat responses

### 2. Test Utilities (new `tests/helpers.py`)

- `setup_dependency_overrides()`: Configure FastAPI dependency overrides
- `cleanup_dependency_overrides()`: Clean up after tests
- `create_mock_response()`: Generate mock responses
- `assert_response_structure()`: Validate response format
- `assert_error_response()`: Validate error responses

### 3. Test Patterns

#### Pattern 1: Async Endpoint Tests
```python
@pytest.mark.asyncio
async def test_endpoint(async_api_client, mock_async_client):
    mock_async_client.method = AsyncMock(return_value=mock_data)
    response = await async_api_client.post("/endpoint", json=data)
    assert response.status_code == 200
```

#### Pattern 2: Sync Endpoint Tests
```python
def test_endpoint(sync_api_client):
    response = sync_api_client.get("/endpoint")
    assert response.status_code == 200
```

#### Pattern 3: Streaming Tests
```python
@pytest.mark.asyncio
async def test_streaming(async_api_client, mock_async_client):
    stream = mock_async_client.generate_stream.return_value
    async for chunk in stream:
        # Verify chunk format
```

## Dependency Injection Testing Strategy

### Problem
FastAPI's `TestClient` doesn't handle async dependencies correctly. It treats `Depends()` parameters as query parameters.

### Solution
1. Use `httpx.AsyncClient` for all async endpoints with dependencies
2. Override all dependencies in the chain (nested dependencies too)
3. Use `app.dependency_overrides` for dependency injection
4. Clean up overrides after each test

### Dependency Override Pattern
```python
# Override all dependencies in chain
app.dependency_overrides[get_client_adapter] = lambda: client_adapter
app.dependency_overrides[get_logger_adapter] = lambda: logger_adapter
app.dependency_overrides[get_metrics_adapter] = lambda: metrics_adapter
app.dependency_overrides[get_use_case] = lambda: use_case
app.dependency_overrides[get_queue] = lambda: queue
```

## Test File Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── helpers.py               # Reusable test utilities (NEW)
├── test_api_server.py       # API endpoint tests
├── test_api_streaming.py    # Streaming endpoint tests
├── test_async_client.py     # Async client tests
├── test_client.py           # Sync client tests
├── test_queue.py            # Queue component tests
├── test_resilience.py       # Resilience component tests
├── test_telemetry.py        # Telemetry component tests
└── ...
```

## Migration Plan

### Phase 1: Create Reusable Components
1. ✅ Create `helpers.py` with utility functions
2. ✅ Consolidate fixtures in `conftest.py`
3. ✅ Create dependency setup/teardown utilities

### Phase 2: Update API Server Tests
1. Convert all async endpoint tests to use `AsyncClient`
2. Update fixtures to use new reusable components
3. Ensure proper cleanup after each test

### Phase 3: Update Other Test Files
1. Update streaming tests
2. Update integration tests
3. Update body parsing tests

### Phase 4: Validation
1. Run full test suite
2. Fix any remaining issues
3. Verify test isolation
4. Check test coverage

## Test Coverage Goals

- **Unit Tests**: 90%+ coverage for domain and application layers
- **Integration Tests**: All API endpoints covered
- **Edge Cases**: All error paths and boundary conditions
- **Streaming**: All streaming scenarios covered

## Best Practices

1. **Naming**: Use descriptive test names that explain what is being tested
2. **Arrange-Act-Assert**: Follow AAA pattern consistently
3. **One Assertion Per Test**: Focus each test on one behavior
4. **Test Data**: Use fixtures for reusable test data
5. **Cleanup**: Always clean up resources (overrides, mocks, etc.)
6. **Isolation**: Tests should not depend on each other
7. **Speed**: Tests should run quickly (< 1 second per test)

## Common Pitfalls to Avoid

1. ❌ Using `TestClient` for async endpoints with dependencies
2. ❌ Not cleaning up dependency overrides
3. ❌ Testing implementation details instead of behavior
4. ❌ Over-mocking internal components
5. ❌ Tests that depend on execution order
6. ❌ Not isolating test state

## Maintenance

- Review and update tests when architecture changes
- Keep fixtures and utilities up to date
- Document any test-specific patterns
- Regular test suite audits

