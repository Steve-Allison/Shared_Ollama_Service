# API Reference Documentation

Complete API reference for the Shared Ollama Service client library.

## Table of Contents

- [Synchronous Client](#synchronous-client)
- [Async Client](#async-client)
- [Resilience Features](#resilience-features)
- [Monitoring & Metrics](#monitoring--metrics)
- [Analytics](#analytics)
- [Utilities](#utilities)

## Synchronous Client

### `SharedOllamaClient`

Main synchronous client for interacting with the Ollama service.

#### Initialization

```python
from shared_ollama_client import SharedOllamaClient, OllamaConfig

# With default configuration
client = SharedOllamaClient()

# With custom configuration
config = OllamaConfig(
    base_url="http://localhost:11434",
    default_model="qwen2.5vl:7b",
    timeout=60,
    verbose=False,
)
client = SharedOllamaClient(config=config)
```

#### Methods

##### `list_models() -> list[dict[str, Any]]`

List all available models.

```python
models = client.list_models()
for model in models:
    print(f"{model['name']}: {model['size']} bytes")
```

##### `generate(prompt, model=None, system=None, options=None, stream=False) -> GenerateResponse`

Generate text using a model.

**Parameters:**
- `prompt` (str): The prompt to generate from
- `model` (str | None): Model name (uses default if None)
- `system` (str | None): Optional system prompt
- `options` (GenerateOptions | None): Generation options
- `stream` (bool): Whether to stream response

**Returns:** `GenerateResponse` with generated text and metadata

**Example:**
```python
from shared_ollama_client import GenerateOptions

options = GenerateOptions(
    temperature=0.7,
    max_tokens=100,
)

response = client.generate(
    "Why is the sky blue?",
    options=options,
)
print(response.text)
```

##### `chat(messages, model=None, stream=False) -> dict[str, Any]`

Chat with the model using conversation format.

**Parameters:**
- `messages` (list[dict[str, str]]): List of messages with "role" and "content"
- `model` (str | None): Model name
- `stream` (bool): Whether to stream response

**Example:**
```python
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's 2+2?"},
]

response = client.chat(messages)
print(response["message"]["content"])
```

##### `health_check() -> bool`

Check if the Ollama service is healthy.

```python
if client.health_check():
    print("Service is healthy")
else:
    print("Service is unavailable")
```

##### `get_model_info(model) -> dict[str, Any] | None`

Get information about a specific model.

```python
info = client.get_model_info("qwen2.5vl:7b")
if info:
    print(f"Model size: {info['size']} bytes")
```

## Async Client

### `AsyncSharedOllamaClient`

Async/await client for modern Python applications.

#### Initialization

```python
import asyncio
from shared_ollama_client_async import AsyncSharedOllamaClient

async def main():
    async with AsyncSharedOllamaClient() as client:
        response = await client.generate("Hello!")
        print(response.text)

asyncio.run(main())
```

#### Methods

All methods are async versions of the synchronous client methods:

- `async def list_models() -> list[dict[str, Any]]`
- `async def generate(...) -> GenerateResponse`
- `async def chat(...) -> dict[str, Any]`
- `async def health_check() -> bool`
- `async def get_model_info(model) -> dict[str, Any] | None`

## Resilience Features

### `ResilientOllamaClient`

Client with automatic retry and circuit breaker.

```python
from resilience import ResilientOllamaClient, RetryConfig, CircuitBreakerConfig

retry_config = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=120.0,
)

circuit_config = CircuitBreakerConfig(
    failure_threshold=10,
    success_threshold=3,
    timeout=120.0,
)

client = ResilientOllamaClient(
    retry_config=retry_config,
    circuit_breaker_config=circuit_config,
)

# Automatically uses retry and circuit breaker
response = client.generate("Hello!")
```

## Monitoring & Metrics

### `MetricsCollector`

Track request metrics and performance.

```python
from monitoring import MetricsCollector, track_request

# Track a request
with track_request("qwen2.5vl:7b", "generate"):
    response = client.generate("Hello!")

# Get metrics
metrics = MetricsCollector.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
print(f"P95 latency: {metrics.p95_latency_ms:.2f}ms")
```

### Metrics Structure

```python
@dataclass
class ServiceMetrics:
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_by_model: dict[str, int]
    requests_by_operation: dict[str, int]
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    errors_by_type: dict[str, int]
    last_request_time: datetime | None
    first_request_time: datetime | None
```

## Analytics

### `AnalyticsCollector`

Enhanced analytics with project-level tracking and time-series analysis.

```python
from analytics import AnalyticsCollector, track_request_with_project

# Track with project identifier
with track_request_with_project(
    "qwen2.5vl:7b",
    "generate",
    project="knowledge_machine",
):
    response = client.generate("Hello!")

# Get analytics
analytics = AnalyticsCollector.get_analytics()
print(f"Requests by project: {analytics.requests_by_project}")

# Export to JSON
AnalyticsCollector.export_json("analytics.json")

# Export to CSV
AnalyticsCollector.export_csv("analytics.csv")
```

### Analytics Dashboard

Use the CLI dashboard to view analytics:

```bash
python scripts/view_analytics.py
python scripts/view_analytics.py --project knowledge_machine
python scripts/view_analytics.py --window 60
python scripts/view_analytics.py --export analytics.json
```

## Utilities

### `get_ollama_base_url() -> str`

Get Ollama base URL from environment or default.

```python
from utils import get_ollama_base_url

url = get_ollama_base_url()
# Checks: OLLAMA_BASE_URL, OLLAMA_HOST/OLLAMA_PORT, defaults to http://localhost:11434
```

### `check_service_health(base_url=None, timeout=5) -> tuple[bool, str | None]`

Check if Ollama service is healthy.

```python
from utils import check_service_health

is_healthy, error = check_service_health()
if not is_healthy:
    print(f"Service unavailable: {error}")
```

### `ensure_service_running(base_url=None, raise_on_fail=True) -> bool`

Ensure service is running, optionally raise exception.

```python
from utils import ensure_service_running

ensure_service_running()  # Raises ConnectionError if not running
```

### `import_client() -> Type[SharedOllamaClient]`

Dynamically import the client class.

```python
from utils import import_client

ClientClass = import_client()
client = ClientClass()
```

## Data Models

### `GenerateOptions`

```python
@dataclass
class GenerateOptions:
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
```

### `GenerateResponse`

```python
@dataclass
class GenerateResponse:
    text: str
    model: str
    context: list[int] | None
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
```

### `Model` Enum

```python
class Model(str, Enum):
    QWEN25_VL_7B = "qwen2.5vl:7b"
    QWEN25_7B = "qwen2.5:7b"
    QWEN25_14B = "qwen2.5:14b"
    GRANITE_4_H_TINY = "granite4:tiny-h"
```

## Error Handling

All client methods raise appropriate exceptions:

- `ConnectionError`: Service is not available
- `requests.HTTPError`: HTTP errors from Ollama API
- `TimeoutError`: Request timeout

Use resilience features for automatic retry:

```python
from resilience import ResilientOllamaClient

client = ResilientOllamaClient()
# Automatically retries on failure
response = client.generate("Hello!")
```

## OpenAPI Specification

Full API specification available in `docs/openapi.yaml`. View interactive documentation:

```bash
# Install Swagger UI (optional)
pip install swagger-ui-py

# Serve documentation
python -m swagger_ui --spec docs/openapi.yaml
```

Or use online tools like [Swagger Editor](https://editor.swagger.io/) to view `docs/openapi.yaml`.

