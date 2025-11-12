# Shared Ollama Service

A centralized Ollama instance for all AI projects to reduce duplication and improve resource management.

**📚 Documentation**: See [docs/README.md](docs/README.md) for complete documentation index.

## Overview

This service provides a single Ollama instance accessible on port `11434` that all projects can use:

- **Architecture snapshot**: see `docs/ARCHITECTURE.md` for diagrams, request flow, and runtime environments.
- **Scaling playbooks**: see `docs/SCALING_AND_LOAD_TESTING.md` for concurrency tuning and load-testing guidance.
- **Knowledge Machine**
- **Course Intelligence Compiler**
- **Story Machine**
- **Docling_Machine**

## Models Available

**Note**: Models are loaded on-demand. Only one model is in memory at a time based on which model is requested.

- **Primary**: `qwen2.5vl:7b` (7B parameters, vision-language model)
  - Vision-language model with multimodal capabilities
  - Excellent performance for vision and text tasks
  - Loaded into memory when requested (~6 GB RAM)
- **Standard**: `qwen2.5:7b` (7B parameters, text-only model)
  - Efficient text-only model with excellent performance
  - Fast inference for text generation tasks
  - Loaded into memory when requested (~4.5 GB RAM)
- **Secondary**: `qwen2.5:14b` (14.8B parameters)
  - Large language model with excellent reasoning
  - Good alternative for text-only tasks
  - Loaded into memory when requested (~9 GB RAM)
- **Granite 4.0**: `granite4:tiny-h` (7B total, 1B active, hybrid MoE)
  - IBM Granite 4.0 H Tiny - Hybrid Mamba/Transformer architecture
  - Optimized for RAG, function calling, and agentic workflows
  - ~70% less RAM for long contexts compared to conventional transformers
  - Up to 128K+ context length (validated to 128K, trained to 512K)
  - Excellent instruction following and tool-calling capabilities
  - Loaded into memory when requested (~8 GB RAM)

Models remain in memory for 5 minutes after last use (OLLAMA_KEEP_ALIVE), then are automatically unloaded to free memory. Switching between models requires a brief load time (~2-3 seconds).

## Project Structure

```
shared_ollama_service/
├── src/shared_ollama/
│   ├── __init__.py                 # Public SDK exports
│   ├── client/
│   │   ├── __init__.py
│   │   ├── sync.py                 # Synchronous client implementation
│   │   └── async_client.py         # Async client (httpx-based)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── utils.py                # Service discovery & helpers
│   │   └── resilience.py           # Circuit breaker, retries, resilient client
│   └── telemetry/
│       ├── __init__.py
│       ├── metrics.py              # Request telemetry & Prometheus-ready helpers
│       ├── analytics.py            # Project-level analytics & exports
│       ├── performance.py          # Structured performance logging
│       └── structured_logging.py   # JSONL request logging
├── tests/                          # Unit tests for the SDK
└── scripts/                        # Operational CLI utilities (load tests, analytics, etc.)
```

All modules are shipped as a package (installable via `pip install -e .`) with type stubs co-located under `src/shared_ollama/**/*.pyi`.

## Installation

### ⚡ Native Installation (Apple Silicon MPS Optimized)

**Best Performance**: Native Ollama with explicit MPS (Metal Performance Shaders) configuration provides maximum GPU acceleration on Apple Silicon.

```bash
# Install and setup native Ollama with MPS optimization
./scripts/install_native.sh

# Service runs manually - start when needed:
ollama serve

# Optional (NOT recommended): Setup as launchd service for automatic startup
# ./scripts/setup_launchd.sh
```

**MPS/Metal Optimizations Enabled:**

- ✅ `OLLAMA_METAL=1` - Explicit Metal/MPS GPU acceleration
- ✅ `OLLAMA_NUM_GPU=-1` - All Metal GPU cores utilized
- ✅ Maximum GPU utilization for fastest inference
- ✅ All 10 CPU cores automatically utilized
- ✅ Lower memory overhead
- ✅ Native macOS integration

### Pull Models

**Option 1: Manual Pull**

```bash
# Pull primary model (qwen2.5vl:7b)
ollama pull qwen2.5vl:7b

# Pull standard model (qwen2.5:7b)
ollama pull qwen2.5:7b

# Pull secondary model (qwen2.5:14b)
ollama pull qwen2.5:14b

# Pull Granite 4.0 H Tiny model (granite4:tiny-h)
ollama pull granite4:tiny-h
```

**Option 2: Automated Pre-download (Recommended)**

```bash
# Pre-download all required models
./scripts/preload_models.sh
```

This automatically checks and downloads all required models, ensuring they're ready before first use.

### Python Development Environment

**Recommended**: Use a virtual environment for development and testing.

```bash
# Create virtual environment (modern best practice: .venv)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies (choose one method):
pip install -r requirements.txt -c constraints.txt
# OR (modern approach, preferred during development):
pip install -e .[dev] -c constraints.txt

# Deactivate when done
deactivate
```

**Dependency Management:**

- `requirements.txt` - Simple dependency list (pip install -r requirements.txt)
- `pyproject.toml` - Modern Python packaging standard (pip install -e .)
- Both files are provided for compatibility

**Install Development Tools:**

```bash
# Install with development dependencies (Ruff, Pyright, pytest)
pip install -e ".[dev]" -c constraints.txt
```

**Modern Development Tools:**

- **Ruff** (v0.14.4+) - Fast linter and formatter (replaces Black, isort, flake8, etc.)
- **Pyright** (v1.1.407+) - Type checker (Microsoft's static type checker)
- **pytest** (v9.0.0+) - Modern testing framework

**Quick Commands:**

**REST API:**
```bash
./scripts/start_api.sh          # Start async REST API server (port 8000)
curl http://localhost:8000/api/v1/health  # Health check
# Visit http://localhost:8000/api/docs for interactive API docs
# Fully async implementation for maximum concurrency
```

**Ollama Service:**

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking
pyright src/shared_ollama

# Run tests
pytest

# Run API server tests
pytest tests/test_api_server.py -v

# Run async load test script (headless)
python scripts/async_load_test.py --requests 200 --workers 20

# Optional: run async load test (requires locust)
locust -f load_test_locust.py --host http://localhost:11434

# Or use Makefile (if available)
make lint        # Run linter
make format      # Format code
make type-check  # Type checking
make test        # Run tests
make check       # Run all checks
make fix         # Auto-fix issues
```

**Why `.venv` instead of `venv`?**

- `.venv` is the modern convention (hidden directory keeps project cleaner)
- Many modern tools (VS Code, PyCharm) auto-detect `.venv` by default
- Both are acceptable, but `.venv` is increasingly preferred in 2024+

**Note for Consuming Projects**: Projects that import this library should use their own virtual environments. The shared library can be imported via `sys.path` without needing to install its dependencies globally.

### Verify Installation

```bash
# Quick status check (recommended)
./scripts/status.sh

# Comprehensive health check
./scripts/health_check.sh

# Or manually check
curl http://localhost:11434/api/tags

# List installed models
ollama list
```

## Quick Start - Using in Your Projects

The easiest way to use the shared service from any project:

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")

from shared_ollama import SharedOllamaClient
from utils import ensure_service_running

# Ensure service is running (with helpful error messages)
ensure_service_running()

# Create client (auto-discovers URL from environment)
client = SharedOllamaClient()

# Use it!
response = client.generate("Hello, world!")
print(response.text)
```

**Environment Variables** (optional - defaults work too):

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
```

**API Documentation**: See [API_REFERENCE.md](docs/API_REFERENCE.md) for complete REST API documentation.

See [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for complete integration instructions and project-specific examples.

## Usage in Projects

### 🚀 Recommended: REST API (Language Agnostic)

**Best for**: All projects (Python, TypeScript, Go, Rust, etc.)

The REST API provides centralized logging, metrics, and rate limiting for all projects. Built with FastAPI and fully async for maximum concurrency and performance.

```bash
# Start the API server (runs on port 8000 by default)
./scripts/start_api.sh
```

**Key Features:**
- ✅ **Fully Async**: Uses `AsyncSharedOllamaClient` for non-blocking I/O operations
- ✅ **High Concurrency**: Handles multiple concurrent requests efficiently
- ✅ **Language Agnostic**: Works with any language that supports HTTP
- ✅ **Centralized Logging**: All requests logged to structured JSON logs
- ✅ **Rate Limiting**: Protects service from overload (60 req/min for generate/chat)
- ✅ **Request Tracking**: Unique request IDs for debugging
- ✅ **Project Identification**: Track usage by project via `X-Project-Name` header
- ✅ **Input Validation**: Automatic validation of prompts, messages, and parameters
- ✅ **Enhanced Error Handling**: Specific HTTP status codes (400, 422, 429, 500, 503, 504) with actionable error messages
- ✅ **Comprehensive Testing**: Full test suite with 20+ test cases covering all endpoints and error scenarios

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "Hello, world!",
        "model": "qwen2.5vl:7b"
    }
)
print(response.json()["text"])
```

**TypeScript/JavaScript Example:**
```typescript
const res = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Hello, world!",
    model: "qwen2.5vl:7b"
  })
});
const data = await res.json();
console.log(data.text);
```

**Go Example:**
```go
resp, err := http.Post(
    "http://localhost:8000/api/v1/generate",
    "application/json",
    bytes.NewBuffer(jsonData),
)
```

**API Documentation**: Visit `http://localhost:8000/api/docs` for interactive API documentation.

**Benefits:**
- ✅ **Fully Async**: Non-blocking I/O operations for maximum concurrency
- ✅ **Language Agnostic**: Works with any language (Python, TypeScript, Go, Rust, etc.)
- ✅ **Streaming Support**: Real-time token-by-token responses for better UX
- ✅ **Request Queue**: Graceful handling of traffic spikes (up to 50 queued requests)
- ✅ **Centralized Logging**: All requests logged to structured JSON logs (`logs/requests.jsonl`)
- ✅ **Unified Metrics**: Aggregated metrics across all projects
- ✅ **Rate Limiting**: 60 requests/minute per IP for generate/chat, 30/min for models
- ✅ **Request Tracking**: Unique request IDs for debugging and support
- ✅ **Project Identification**: Track usage by project via `X-Project-Name` header
- ✅ **High Performance**: Async implementation handles concurrent requests efficiently
- ✅ **Input Validation**: Automatic validation prevents invalid requests (empty prompts, invalid roles, length limits)
- ✅ **Robust Error Handling**: Specific error responses for validation errors (422), rate limits (429), service unavailable (503), timeouts (504), and more
- ✅ **Production Ready**: Comprehensive error handling, input validation, and test coverage

### Streaming Responses

The REST API supports real-time streaming for token-by-token responses, providing better user experience for long-running AI generations.

**Python Client Streaming:**
```python
import asyncio
from shared_ollama import AsyncSharedOllamaClient

async def stream_example():
    async with AsyncSharedOllamaClient() as client:
        # Stream text generation
        async for chunk in client.generate_stream(prompt="Write a story about AI"):
            print(chunk["chunk"], end="", flush=True)
            if chunk["done"]:
                print(f"\n\nCompleted in {chunk['latency_ms']}ms")

        # Stream chat completion
        messages = [{"role": "user", "content": "Tell me a joke"}]
        async for chunk in client.chat_stream(messages=messages):
            print(chunk["chunk"], end="", flush=True)
            if chunk["done"]:
                print(f"\n\nModel: {chunk['model']}, Latency: {chunk['latency_ms']}ms")

asyncio.run(stream_example())
```

**REST API Streaming:**
```python
import requests

# Streaming generate endpoint
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={"prompt": "Write a story", "model": "qwen2.5:7b", "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        # Parse Server-Sent Events format
        if line.startswith(b'data: '):
            import json
            chunk = json.loads(line[6:])  # Remove "data: " prefix
            print(chunk['chunk'], end='', flush=True)
            if chunk['done']:
                print(f"\n\nLatency: {chunk['latency_ms']}ms")
```

**TypeScript/JavaScript Streaming:**
```typescript
const response = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Write a story",
    model: "qwen2.5:7b",
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const chunk = JSON.parse(line.slice(6));
      process.stdout.write(chunk.chunk);
      if (chunk.done) {
        console.log(`\n\nLatency: ${chunk.latency_ms}ms`);
      }
    }
  }
}
```

**Streaming Format:**

- Uses Server-Sent Events (SSE) format: `data: {json}\n\n`
- Each chunk contains: `chunk` (text), `done` (boolean), `model`, `request_id`
- Final chunk (when `done=True`) includes metrics: `latency_ms`, `model_load_ms`, `model_warm_start`, token counts

### Request Queue Management

The REST API includes intelligent request queuing to handle traffic spikes gracefully. Instead of immediately rejecting requests with 503 errors when capacity is reached, the queue system allows requests to wait for available slots.

**Configuration:**

- **Max Concurrent**: 3 requests processed simultaneously
- **Max Queue Size**: 50 requests can wait in queue
- **Default Timeout**: 60 seconds wait time before timeout

**How it Works:**

1. When all 3 processing slots are busy, new requests enter the queue
2. Requests wait for an available slot (up to 60 seconds by default)
3. Once a slot opens, the next queued request begins processing
4. If the queue fills (50+ waiting), new requests are rejected with clear error messages

**Queue Statistics Endpoint:**

Monitor queue health in real-time:

```bash
curl http://localhost:8000/api/v1/queue/stats
```

Response:

```json
{
  "queued": 2,              // Currently waiting
  "in_progress": 3,         // Currently processing
  "completed": 145,         // Total completed
  "failed": 2,              // Total failed
  "rejected": 0,            // Rejected (queue full)
  "timeout": 0,             // Timed out in queue
  "total_wait_time_ms": 15420.5,  // Total wait time
  "max_wait_time_ms": 3250.2,     // Maximum wait observed
  "avg_wait_time_ms": 106.3,      // Average wait time
  "max_concurrent": 3,      // Config: max concurrent
  "max_queue_size": 50,     // Config: max queue size
  "default_timeout": 60.0   // Config: timeout (seconds)
}
```

**Benefits:**

- ✅ **Graceful Degradation**: Requests wait instead of failing immediately
- ✅ **Better UX**: Users see progress instead of errors during traffic spikes
- ✅ **Visibility**: Real-time queue metrics for monitoring
- ✅ **Automatic Cleanup**: Failed requests don't block the queue
- ✅ **Timeout Protection**: Prevents indefinite waiting

**Example Monitoring Script:**

```python
import requests
import time

def monitor_queue():
    while True:
        response = requests.get("http://localhost:8000/api/v1/queue/stats")
        stats = response.json()

        print(f"Queue Status: {stats['queued']} waiting, "
              f"{stats['in_progress']} processing, "
              f"avg wait: {stats['avg_wait_time_ms']:.1f}ms")

        if stats['rejected'] > 0:
            print(f"⚠️  {stats['rejected']} requests rejected (queue full)")

        time.sleep(5)

monitor_queue()
```

### Alternative: Using the Shared Client Library (Python Only)

**Best for**: Python projects that want direct library access

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")

from shared_ollama import SharedOllamaClient
from shared_ollama.core.utils import ensure_service_running

# Automatic service discovery from environment
ensure_service_running()

client = SharedOllamaClient()
response = client.generate("Your prompt here")
```

### Project-Specific Integration

#### Knowledge Machine

Update `Knowledge_Machine/config/main.py`:

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")
from shared_ollama import OllamaConfig, SharedOllamaClient

# Or configure via environment
import os
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
default_model = "qwen2.5vl:7b"

client = SharedOllamaClient(OllamaConfig(
    base_url=ollama_base_url,
    default_model=default_model
))
```

#### Course Intelligence Compiler

Update `Course_Intelligence_Compiler/config/rag_config.yaml`:

```yaml
generation:
  ollama:
    base_url: "http://localhost:11434"  # Or use OLLAMA_BASE_URL env var
    model: "qwen2.5vl:7b"  # or "qwen2.5:14b" for alternative model
```

#### Story Machine

Update `Story_Machine/src/story_machine/core/config.py`:

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")
from shared_ollama import SharedOllamaClient
from shared_ollama.core.utils import get_ollama_base_url

# Auto-discover from environment
base_url = get_ollama_base_url()
model = "qwen2.5vl:7b"

client = SharedOllamaClient()
```

See [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for detailed project-specific examples and migration instructions.

## Configuration

### Environment Variables

**For Client Projects** (detected automatically by `shared_ollama.core.utils`):

```bash
export OLLAMA_BASE_URL="http://localhost:11434"  # Preferred
# OR
export OLLAMA_HOST="localhost"
export OLLAMA_PORT="11434"
```

**For Service Configuration** (create `.env` file in project root):

```env
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
OLLAMA_KEEP_ALIVE=5m  # How long models stay loaded after last use (see Keep-Alive section below)
OLLAMA_DEBUG=false
```

The client automatically discovers the service URL from these environment variables, so projects don't need hardcoded URLs.

### Keep-Alive Configuration

The `OLLAMA_KEEP_ALIVE` setting determines how long models remain in memory after the last request before being automatically unloaded.

**Trade-offs:**

| Keep-Alive Time | Pros | Cons | Best For |
|----------------|------|------|----------|
| **1-2 minutes** | Frees memory quickly, allows faster model switching | More frequent reloads, slower responses after idle periods | Memory-constrained environments, infrequent use |
| **5 minutes** (current) | Balanced - reasonable response time without excessive memory use | Model unloaded after 5 min idle | **General use, development environments** |
| **15-30 minutes** | Fewer reloads, faster responses for active sessions | Models stay in memory longer, blocks other models | Active development, frequent model switching |
| **Infinite** (`-1` or `0`) | Never unloads, fastest response time | Always uses memory, blocks other models | Production with dedicated models, high-traffic |

**Recommendations:**

- **Development/Testing**: `5m` (current) - Good balance for intermittent use
- **Active Development**: `15m-30m` - Reduces reloads during active coding sessions
- **Memory-Constrained**: `2m-3m` - Faster memory release
- **Production (Single Model)**: `15m-30m` or `0` - Keep model hot for consistent performance
- **Production (Multiple Models)**: `5m-10m` - Balance between performance and memory efficiency

**To change keep-alive:**

Set the environment variable when starting Ollama:

```bash
export OLLAMA_KEEP_ALIVE=15m
ollama serve
```

Or set it when starting Ollama manually (launchd not recommended)

### Port Configuration

- **Default Port**: 11434 (standard Ollama port)
- **API Endpoint**: `http://localhost:11434/api/generate`
- **Tags Endpoint**: `http://localhost:11434/api/tags`

## Health Checks

### Automated Health Check

```bash
./scripts/health_check.sh
```

This checks:

- Ollama service is running
- All models are available
- API is responding

### Manual Testing

```bash
# List models
curl http://localhost:11434/api/tags

# Test generation
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5vl:7b",
  "prompt": "Why is the sky blue?"
}'
```

## Model Management

### List Loaded Models

```bash
ollama list
```

### Remove a Model

```bash
ollama rm model_name
```

### Update Models

```bash
# Pull latest version
ollama pull qwen2.5vl:7b
```

## Service Management

**Default: Manual Start** - Service must be started manually when needed.

```bash
# Start service with MPS/Metal optimization (RECOMMENDED)
./scripts/start.sh

# Or start manually with optimizations:
export OLLAMA_METAL=1
export OLLAMA_NUM_GPU=-1
ollama serve > ./logs/ollama.log 2> ./logs/ollama.error.log &

# Stop service
pkill ollama
# Or use shutdown script
./scripts/shutdown.sh

# Check if running
curl http://localhost:11434/api/tags

# View logs
tail -f ./logs/ollama.log
```

**Optional: Auto-start on Boot** (NOT recommended - disabled by default)

```bash
# Only if you want the service to auto-start on login/boot:
# ./scripts/setup_launchd.sh

# Manage launchd service (if enabled)
# launchctl load ~/Library/LaunchAgents/com.ollama.service.plist  # Start
# launchctl unload ~/Library/LaunchAgents/com.ollama.service.plist  # Stop
# launchctl list | grep ollama  # Status
```

## Performance Optimizations

### Current Optimizations

✅ **Fully Optimized for Apple Silicon MPS (Metal Performance Shaders):**

- ✅ **MPS/Metal GPU**: Explicitly enabled (`OLLAMA_METAL=1`)
- ✅ **GPU Cores**: All Metal GPU cores utilized (`OLLAMA_NUM_GPU=-1`)
- ✅ **CPU**: All 10 cores automatically detected and utilized
- ✅ **Threading**: Auto-detected based on CPU cores
- ✅ **Memory**: Efficient allocation for both models simultaneously
- ✅ **Performance**: Maximum GPU acceleration via Metal/MPS

### Hardware Detection

The service auto-detects hardware:

- **Apple Silicon** (M1/M2/M3/M4): Metal/MPS GPU acceleration (explicitly enabled)
- **CPU**: All cores automatically detected and utilized (10 cores)
- **Memory**: Efficient allocation based on model requirements
- **GPU**: All Metal GPU cores available for inference

### Apple Silicon MPS Optimization

**Explicit MPS Configuration:**

- ✅ `OLLAMA_METAL=1`: Explicitly enables Metal/MPS acceleration
- ✅ `OLLAMA_NUM_GPU=-1`: Uses all available Metal GPU cores
- ✅ `OLLAMA_NUM_THREAD`: Auto-detected (matches 10 CPU cores)

**Performance Benefits:**

- Maximum GPU utilization via Metal Performance Shaders
- Faster inference times with full GPU acceleration
- Efficient memory management for concurrent model execution
- Optimal resource allocation for Apple Silicon architecture

**To verify MPS is active:**

```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep -i metal

# Monitor GPU usage (in Activity Monitor or with system_profiler)
```

## Troubleshooting

### Service Won't Start

```bash
# Check if Ollama is installed
which ollama

# Start service manually
ollama serve

# Check if port is in use
lsof -i :11434

# View logs (if using launchd)
tail -f ~/.ollama/ollama.log
```

### Models Not Found

```bash
# Pull models
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
ollama pull granite4:tiny-h

# Verify
ollama list
```

### Connection Refused

```bash
# Check if port is in use
lsof -i :11434

# Kill existing Ollama instance
kill $(lsof -t -i:11434)

# Restart service
ollama serve
```

## Migration Guide

### From Individual Project Ollama Instances

1. **Stop existing instances**

   ```bash
   # Stop any running Ollama processes
   pkill ollama
   ```

2. **Start shared service**

   ```bash
   cd Shared_Ollama_Service
   ./scripts/install_native.sh

   # Or if already installed
   ollama serve
   ```

3. **Update project configurations** (see Usage section above)

4. **Test each project**

   ```bash
   # Test Knowledge Machine
   cd Knowledge_Machine && python -m pytest tests/integration/test_rag_integration.py

   # Test Course Intelligence Compiler
   cd Course_Intelligence_Compiler && python -m pytest tests/common/llm/test_ollama_client.py

   # Test Story Machine
   cd Story_Machine && pytest tests/
   ```

## Security

### Network Isolation

The Ollama service is **not exposed to the internet** by default. Only localhost connections are allowed.

### Model Access Control

Models are stored locally: `~/.ollama/models`

## Monitoring

### Quick Status Check

```bash
# Fast status overview (recommended)
./scripts/status.sh
```

Shows:

- Service health
- Available models
- Process information
- Memory usage
- Quick health test

### Logs

```bash
# View logs (if using launchd service)
tail -f ./logs/ollama.log

# View error logs
tail -f ./logs/ollama.error.log

# View structured request log (JSON lines)
tail -f ./logs/requests.jsonl

# If running manually, logs output to terminal
ollama serve
```

### Metrics & Performance

**Request Metrics** (via monitoring):

- Overall latency (p50, p95, p99)
- Success/failure rates
- Usage by model and operation

**Ollama Service Logs**:

- HTTP request logs: `logs/ollama.log`
- Error logs: `logs/ollama.error.log`
- Performance logs: `logs/performance.jsonl` (if using performance tracking)
- Structured request events (model timings, load durations): `logs/requests.jsonl`

**Performance Analysis**:

```bash
# View performance report
python scripts/performance_report.py

# Filter by model
python scripts/performance_report.py --model qwen2.5vl:7b

# Last hour
python scripts/performance_report.py --window 60
```

**Quick Monitoring**:

- **Quick status**: `./scripts/status.sh` (fast overview)
- **Health checks**: `./scripts/health_check.sh` (comprehensive)
- **Model status**: `curl http://localhost:11434/api/tags`
- **Resource usage**: `top -pid $(pgrep ollama)` or Activity Monitor

**Note**: See `PERFORMANCE_MONITORING.md` for detailed performance tracking capabilities.

## Cost and Resource Management

### Memory Usage

**Memory Usage:**

- `qwen2.5vl:7b`: ~6 GB RAM when loaded
- `qwen2.5:7b`: ~4.5 GB RAM when loaded
- `qwen2.5:14b`: ~9 GB RAM when loaded
- `granite4:tiny-h`: ~8 GB RAM when loaded (but ~70% less RAM for long contexts)
- **Models can run simultaneously** if you have sufficient RAM

**Behavior**: Models are automatically loaded when requested and unloaded after 5 minutes of inactivity. Both models can be active at the same time if needed, reducing switching delays.

### Performance

**Without Warm-up:**

- **First request**: ~2-3 seconds (model loading into memory)
- **Subsequent requests**: ~100-500ms (depends on prompt length)

**With Warm-up (Models Pre-loaded):**

- **First request**: ~100-500ms (model already in memory)
- **Subsequent requests**: ~100-500ms (consistently fast)

See [Warm-up & Pre-loading](#warm-up--pre-loading) section below for setup.

## Warm-up & Pre-loading

### Pre-download Models

Ensure all models are downloaded locally before first use:

```bash
# Pre-download all required models
./scripts/preload_models.sh
```

This verifies models are available locally, eliminating download delays during inference.

### Warm-up Models (Pre-load into Memory)

Pre-load models into memory to eliminate first-request latency:

```bash
# Warm up models (loads them into memory)
./scripts/warmup_models.sh

# Or warm up with custom keep-alive duration
KEEP_ALIVE=60m ./scripts/warmup_models.sh
```

**What this does:**

- Sends minimal requests to each model
- Loads models into GPU/CPU memory
- Sets keep-alive to keep models loaded (default: 30 minutes)
- Reduces first-request latency from ~2-3s to ~100-500ms

### Automated Warm-up on Startup

**Option 1: Manual warm-up after service start**

```bash
# Start Ollama
ollama serve

# In another terminal, warm up models
./scripts/warmup_models.sh
```

**Option 2: Cron job for warm-up** (if using launchd service)

```bash
# Add to crontab to warm up 2 minutes after login
crontab -e

# Add this line:
@reboot sleep 120 && /path/to/Shared_Ollama_Service/scripts/warmup_models.sh
```

**Option 3: Warm-up via API call**

```bash
# Warm up a specific model
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5vl:7b",
  "prompt": "Hi",
  "options": {"num_predict": 1},
  "keep_alive": "30m"
}'
```

### Keep-Alive Recommendations for Production

**High-Traffic Production:**

```bash
# Keep models loaded indefinitely
KEEP_ALIVE=-1 ./scripts/warmup_models.sh
```

**Development/Testing:**

```bash
# Keep models loaded for active session
KEEP_ALIVE=30m ./scripts/warmup_models.sh
```

**Memory-Constrained:**

```bash
# Shorter keep-alive, models reload as needed
KEEP_ALIVE=10m ./scripts/warmup_models.sh
```

### Performance Comparison

| Setup | First Request | Subsequent Requests | Best For |
|-------|--------------|---------------------|----------|
| **No warm-up** | 2-3 seconds | 100-500ms | Development, occasional use |
| **Warm-up (30m keep-alive)** | 100-500ms | 100-500ms | Active development |
| **Warm-up (infinite keep-alive)** | 100-500ms | 100-500ms | Production, high-traffic |

## New Features & Enhancements

### Async/Await Support

Modern async/await client for asynchronous Python applications:

```python
import asyncio
from shared_ollama import AsyncSharedOllamaClient

async def main():
    async with AsyncSharedOllamaClient() as client:
        response = await client.generate("Hello!")
        print(response.text)

asyncio.run(main())
```

**Installation**: `pip install -e ".[async]"`

### Monitoring & Metrics

Track usage, latency, and errors across all projects:

```python
from shared_ollama import MetricsCollector, track_request

# Track a request
with track_request("qwen2.5vl:7b", "generate"):
    response = client.generate("Hello!")

# Get metrics
metrics = MetricsCollector.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
```

### Enhanced Resilience

Automatic retry with exponential backoff and circuit breaker:

```python
from resilience import ResilientOllamaClient

client = ResilientOllamaClient()
response = client.generate("Hello!")  # Automatic retry & circuit breaker
```

### Testing

Comprehensive test suite for reliability:

```bash
# Run tests
pytest

# With coverage
pytest --cov
```

See `IMPLEMENTED_ENHANCEMENTS.md` for full details.

### Enhanced Analytics

Advanced analytics with project-level tracking and time-series analysis:

```python
from shared_ollama import AnalyticsCollector, track_request_with_project

# Track with project identifier
with track_request_with_project("qwen2.5vl:7b", "generate", project="knowledge_machine"):
    response = client.generate("Hello!")

# View analytics dashboard
python scripts/view_analytics.py

# Export analytics
AnalyticsCollector.export_json("analytics.json")
AnalyticsCollector.export_csv("analytics.csv")
```

### API Documentation

Complete API reference and OpenAPI specification:

- **API Reference**: See `docs/API_REFERENCE.md`
- **OpenAPI Spec**: `docs/openapi.yaml` (OpenAPI 3.1.0)
- **View Interactive Docs**: Use Swagger UI or online editor

### Type Stubs

Full type stubs for better IDE support:

```bash
# Type stubs are automatically included
# Your IDE should detect them automatically
```

### CI/CD

GitHub Actions workflows for automated testing and releases:

- **CI Workflow**: `.github/workflows/ci.yml` - Tests, linting, type checking
- **Release Workflow**: `.github/workflows/release.yml` - Automated releases

## Code Quality & Modernization

### Python 3.13+ Modernization

This codebase has been fully modernized to leverage Python 3.13+ native features:

**Native Features:**
§s
- ✅ `datetime.now(UTC)` for timezone-aware timestamps
- ✅ `time.perf_counter()` for precise performance measurements
- ✅ Modern type hints with `|` union syntax
- ✅ `collections.abc.Generator` for context managers

**Code Quality:**

- ✅ 90%+ type hint coverage across all modules
- ✅ Comprehensive error handling with specific exception types
- ✅ JSON validation and response structure checking
- ✅ File I/O error handling with detailed logging
- ✅ DRY principles with extracted helper functions

**Testing:**

- ✅ 54 comprehensive tests (all passing)
- ✅ Edge case handling and validation
- ✅ Error scenario testing
- ✅ Performance and resilience testing

See recent commits for detailed modernization improvements.

## Development

This project uses modern Python 3.13+ tooling for development:

### Development Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Code Quality Tools

**Ruff** - Fast linter and formatter (replaces Black, isort, flake8, etc.)

- **Format code**: `ruff format .`
- **Lint code**: `ruff check .`
- **Auto-fix**: `ruff check --fix .`

**Pyright** - Static type checker (Microsoft's type checker)

- **Type check**: `pyright src/shared_ollama`

**Testing** - pytest with coverage

- **Run tests**: `pytest`
- **With coverage**: `pytest --cov`

### Makefile Commands

```bash
make lint        # Run Ruff linter
make format      # Format code with Ruff
make type-check  # Run Pyright type checker
make test        # Run tests with pytest
make check       # Run all checks (lint, format, type-check)
make fix         # Auto-fix linting issues
make all         # Clean, install, format, fix, type-check, test
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run Ruff and Pyright before each commit.

### Configuration Files

- **`pyproject.toml`** - Project configuration, dependencies, Ruff, and Pyright settings
- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`Makefile`** - Convenient development commands

### Python Version

This project requires **Python 3.13+** and uses modern Python features:

- Native type annotations (`list` instead of `List`, `dict` instead of `Dict`)
- Union types (`X | None` instead of `Optional[X]`)
- Modern enum patterns
- Latest typing features

## Contributing

When adding new models or modifying the service:

1. Update installation scripts if needed
2. Update preload/warmup scripts with new models
3. Update this README
4. Test with all projects
5. Document model size and use cases

## License

MIT

## Support

For issues or questions:

- Check logs: `tail -f ~/.ollama/ollama.log`
- Run health check: `./scripts/health_check.sh`
- Verify service: `curl http://localhost:11434/api/tags`
- Open issue in project repository
