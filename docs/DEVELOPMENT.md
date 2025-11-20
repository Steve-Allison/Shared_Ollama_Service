# Development Guide

This guide covers development practices, testing strategies, and contribution guidelines for the Shared Ollama Service.

## Development Setup

### Prerequisites

- Python 3.13+
- Ollama installed and accessible
- Git

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd Shared_Ollama_Service

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]" -c constraints.txt

# Verify setup
./scripts/verify_setup.sh
```

### Configuration

- Model defaults are auto-selected from `config/model_profiles.yaml`
- No `.env` file is required; overrides can be exported directly in your shell
- See [CONFIGURATION.md](CONFIGURATION.md) for all available options

## Testing

### Test Structure

The test suite follows Clean Architecture principles:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **API Tests**: Test HTTP endpoints end-to-end

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/shared_ollama --cov-report=html

# Run specific test file
pytest tests/test_api_server.py

# Run specific test
pytest tests/test_api_server.py::test_generate_endpoint

# Run with verbose output
pytest -v

# Run async tests only
pytest -m asyncio
```

### Test Organization

```
tests/
├── __init__.py
├── conftest.py          # Pytest fixtures and configuration
├── helpers.py           # Reusable test utilities
├── test_api_server.py   # API endpoint tests
├── test_client.py       # Client library tests
├── test_async_client.py # Async client tests
├── test_resilience.py   # Resilience pattern tests
├── test_telemetry.py    # Telemetry tests
├── test_utils.py        # Utility function tests
└── test_queue.py        # Request queue tests
```

### Test Fixtures

Common fixtures available in `conftest.py`:

- `mock_async_client`: Mock AsyncSharedOllamaClient
- `mock_use_cases`: Pre-configured use cases with mocked client
- `test_dependencies`: Complete dependency setup (adapters, queue)
- `async_api_client`: AsyncClient with dependency overrides
- `sync_api_client`: TestClient for sync endpoints

### Writing Tests

#### Async Endpoint Tests

```python
@pytest.mark.asyncio
async def test_endpoint(async_api_client, mock_async_client):
    mock_async_client.method = AsyncMock(return_value=mock_data)
    response = await async_api_client.post("/endpoint", json=data)
    assert response.status_code == 200
```

#### Sync Endpoint Tests

```python
def test_endpoint(sync_api_client):
    response = sync_api_client.get("/endpoint")
    assert response.status_code == 200
```

#### Streaming Tests

```python
@pytest.mark.asyncio
async def test_streaming(async_api_client):
    async with async_api_client.stream("POST", "/endpoint", json=data) as response:
        async for chunk in response.aiter_lines():
            # Process chunk
            pass
```

### Test Coverage

The project aims for comprehensive test coverage:

- All API endpoints tested
- Success and error cases covered
- Edge cases and boundary conditions tested
- Integration tests for critical paths

## Code Style

### Linting

The project uses `ruff` for linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

### Type Checking

The project uses `pyright` for type checking:

```bash
# Type check
pyright src/
```

### Pre-commit Checks

Run before committing:

```bash
# Run all checks
./scripts/ci_check.sh

# Or individually
ruff check src/
pyright src/
pytest
```

## Architecture Guidelines

### Layer Rules

1. **Domain Layer**: No external dependencies, pure business logic
2. **Application Layer**: Depends only on domain, defines interfaces
3. **Interface Adapters**: Depends on application, adapts to frameworks
4. **Infrastructure**: Implements application interfaces

### Dependency Injection

- Use FastAPI's `Depends()` for all dependencies
- Define interfaces using Python `Protocol`
- Avoid global state
- Make dependencies explicit in function signatures

### Adding New Features

1. **Domain**: Add entities/value objects in `domain/`
2. **Application**: Add use cases in `application/`
3. **Interfaces**: Define protocols in `application/interfaces.py`
4. **Infrastructure**: Implement adapters in `infrastructure/`
5. **API**: Add routes in `api/routes/`
6. **Tests**: Add tests in `tests/`

## Debugging

### Local Development

```bash
# Start API server with auto-reload
API_RELOAD=true ./scripts/start.sh

# Or use uvicorn directly
uvicorn shared_ollama.api.server:app --reload --host 0.0.0.0 --port 8000
```

### Logging

Logs are written to:
- `logs/ollama.log` - Ollama service logs
- `logs/ollama.error.log` - Ollama error logs
- `logs/requests.jsonl` - Structured request logs
- `logs/performance.jsonl` - Performance metrics

### Debugging Tips

1. Enable verbose logging: `CLIENT_VERBOSE=true`
2. Check queue statistics: `GET /api/v1/queue/stats`
3. Monitor metrics: `GET /api/v1/metrics`
4. View performance stats: `GET /api/v1/performance/stats`

## Performance Testing

### Load Testing

See [SCALING_AND_LOAD_TESTING.md](SCALING_AND_LOAD_TESTING.md) for detailed load testing guide.

```bash
# Run async load test
python scripts/async_load_test.py --workers 10 --requests 100
```

### Performance Monitoring

```bash
# Generate performance report
python scripts/performance_report.py

# View analytics
python scripts/view_analytics.py
```

## Contributing

### Pull Request Process

1. Create a feature branch
2. Make changes following architecture guidelines
3. Add tests for new features
4. Update documentation as needed
5. Run all checks: `./scripts/ci_check.sh`
6. Submit pull request with clear description

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### Code Review

- All PRs require review
- Tests must pass
- Code must follow style guidelines
- Documentation must be updated

## Troubleshooting

### Common Issues

**Issue**: Tests fail with dependency injection errors
- **Solution**: Ensure using `async_api_client` fixture for async endpoints

**Issue**: Configuration not loading
- **Solution**: Re-run `./scripts/generate_optimal_config.sh` to confirm the detected profile or export overrides in your shell

**Issue**: Type checking errors
- **Solution**: Some false positives may require `# type: ignore` comments

**Issue**: Import errors
- **Solution**: Ensure virtual environment is activated and dependencies installed

## Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [Configuration Guide](CONFIGURATION.md)
- [API Reference](API_REFERENCE.md)
- [Testing Plan](TESTING_PLAN.md) (detailed testing strategy)

