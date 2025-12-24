# Shared Ollama Service

**REST API** - Centralized Ollama service with VLM support (dual format), batch processing, and image compression.

**Key Features**:

- **REST API**: FastAPI-based service (port 8000) that manages Ollama internally
- **Native Ollama Format**: Simple, efficient, direct integration
- **OpenAI-Compatible Format**: For Docling and other OpenAI-compatible clients
- **Automatic Management**: REST API automatically starts and manages Ollama (no manual setup needed)
- **Embeddings API**: Generate vector embeddings for semantic search and RAG systems
- **Response Caching**: Intelligent caching with semantic similarity matching for faster responses
- **Model Management**: Create, copy, and manage custom models via API
- **Automatic Memory Management**: Background service automatically unloads idle models
- **Fine-tuning Helpers**: Scripts and tools for local model customization
- **Agent System**: Support for Ollama 0.13.5+ agent framework

**📚 Documentation**: See [docs/README.md](docs/README.md) for complete documentation index.  
**🛠️ Stability Plan**: See [docs/STABILITY_PLAN.md](docs/STABILITY_PLAN.md) for the hardening roadmap.

## Core Stack (Nov 2025)

- `fastapi` **0.122.0** + `uvicorn[standard]` **0.38.0** for the HTTP layer
- `gunicorn` **23.0.0** for production process supervision
- `psutil` **7.1.3**, `tenacity` **9.1.2**, `cachetools` **6.2.2** for process control, retries, and caching
- `Pillow` **12.0.0** for VLM image handling
- Tooling: `pytest` **9.0.1**, `pytest-asyncio` **1.3.0**, `pytest-cov` **7.0.0**, `ruff` **0.14.6**

See `pyproject.toml` and `constraints.txt` for the authoritative list of pinned versions.

## Overview

This service provides a REST API (port 8000) that manages Ollama internally and makes it accessible to all projects:

- **Architecture snapshot**: see `docs/ARCHITECTURE.md` for diagrams, request flow, and runtime environments.
- **Clean Architecture**: see `docs/CLEAN_ARCHITECTURE_REFACTORING.md` for layer structure and dependency rules.
- **Testing strategy**: see `docs/TESTING_PLAN.md` for comprehensive testing approach and reusable components.
- **Scaling playbooks**: see `docs/SCALING_AND_LOAD_TESTING.md` for concurrency tuning and load-testing guidance.
- **Knowledge Machine**
- **Course Intelligence Compiler**
- **Story Machine**
- **Docling_Machine**

## 5-Minute Quickstart

Get up and running with the Shared Ollama Service in 5 minutes:

### 1. Start the Service

```bash
# Start REST API (automatically manages Ollama)
./scripts/core/start.sh
```

The service will:

- Auto-detect your hardware and configure optimal settings
- Start Ollama with MPS/Metal GPU acceleration (Apple Silicon)
- Launch the REST API on port 8000

### 2. Verify It's Running

```bash
# Health check
curl http://0.0.0.0:8000/api/v1/health

# List available models
curl http://0.0.0.0:8000/api/v1/models

# Inspect the active hardware profile and model recommendations
curl http://0.0.0.0:8000/api/v1/system/model-profile
```

Expected response:

```json
{"status": "healthy", "ollama_status": "running"}
```

### 3. Send Your First Request

**Text Chat (Native Ollama Format):**

```bash
curl -X POST http://0.0.0.0:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:14b-q4_K_M",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence"}
    ]
  }'
```

**Text Chat (OpenAI-Compatible Format):**

```bash
curl -X POST http://0.0.0.0:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:14b-q4_K_M",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence"}
    ]
  }'
```

**Vision-Language Model (with Image):**

```bash
# Encode an image to base64 (replace with your image path)
IMAGE_DATA=$(python3 -c "import base64; print('data:image/jpeg;base64,' + base64.b64encode(open('photo.jpg', 'rb').read()).decode())")

curl -X POST http://0.0.0.0:8000/api/v1/vlm \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"qwen3-vl:8b-instruct-q4_K_M\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What's in this image?\"}],
    \"images\": [\"$IMAGE_DATA\"]
  }"
```

**Generate Embeddings:**

```bash
curl -X POST http://0.0.0.0:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:14b-q4_K_M",
    "prompt": "What is machine learning?"
  }'
```

**Model Management:**

```bash
# List running models
curl http://0.0.0.0:8000/api/v1/models/ps

# Get detailed model information
curl http://0.0.0.0:8000/api/v1/models/qwen3:14b-q4_K_M/show

# Create a custom model from Modelfile
curl -X POST http://0.0.0.0:8000/api/v1/models/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom-model",
    "modelfile": "FROM qwen3:14b-q4_K_M\nSYSTEM \"You are a helpful assistant.\""
  }'
```

### 4. View Interactive API Documentation

Open in your browser:

```text
http://0.0.0.0:8000/api/docs
```

### Next Steps

- **Client Examples**: See [docs/CLIENT_GUIDE.md](docs/CLIENT_GUIDE.md) for Python, TypeScript, and Go examples
- **VLM Guide**: See [docs/VLM_GUIDE.md](docs/VLM_GUIDE.md) for complete vision-language model documentation
- **API Reference**: See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete endpoint documentation
- **Integration**: See [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for project integration examples

## Client Usage Guidelines (RAG & Chat)

The service is multi-tenant and backed by finite Ollama workers. Follow these
guardrails so your workloads stay fast and predictable:

### Request Size & Time Limits
- **Max prompt tokens:** 4 096 (matches current Ollama profile). Trim history or
  summarize before sending. Requests exceeding the limit are rejected with
  `400`/`code=prompt_too_large`.
- **Max request body:** 1.5 MiB (after base64). Compress or split large images.
- **Timeouts:** Text chat endpoints hard-stop at **120 s**, VLM at **150 s**.
  When a timeout hits you will receive `503` with `code=request_timeout`.

### Concurrency & Backpressure
- **Per-tenant concurrency:** 6 in-flight text jobs, 3 VLM jobs. Extra requests
  receive `429` with `Retry-After` headers—respect them.
- **Queue depth:** When the shared queue is full we fail fast with `503
  code=queue_full`. Back off exponentially (≥2 s with jitter) before retrying.
- **Streaming encouraged:** Request streamed responses (`stream=true` in the
  OpenAI format) so you can stop when you have enough tokens and free capacity.

### Payload Hygiene
- Include only the *minimal* retrieved chunks needed for the answer. Use your
  RAG retriever to deduplicate and summarize source docs.
- Drop verbose system prompts; reuse the shared templates from `examples/`.
- For multi-turn chats keep the last 4–6 turns, summarize older context in your
  app, and prepend the summary instead of raw history.

### Recommended Client Behavior
- Propagate `X-Shared-Ollama-Request-Id` into your logs for support.
- Implement cancellation hooks: if the caller disconnects, cancel the HTTP
  request so the server frees the slot immediately.
- Handle structured errors: every error response includes `code` and
  `retry_after` (if applicable). Use those fields rather than guessing.

### Troubleshooting Checklist
1. Capture `X-Shared-Ollama-Request-Id` and search it in `logs/api.log`.
2. `prompt_too_large` → shrink payload; call `/api/v1/system/model-profile`
   for live limits.
3. `queue_full` or `request_timeout` → respect `Retry-After`, stagger retries,
   and consider lowering client-side concurrency.
4. If issues persist, capture logs plus request metadata and open a ticket.

> Tip: [docs/STABILITY_PLAN.md](docs/STABILITY_PLAN.md) tracks planned limit
> changes. Watch that file (or release notes) when upgrading clients.

### Error Codes You Will See
- `prompt_too_large` (400): prompt history exceeded ~4 096 tokens. Trim/summarize
  before retrying.
- `request_too_large` (413): JSON body exceeded **1.5 MiB**. Chunk or compress.
- `queue_full` (503): shared queue saturated. Honor `Retry-After` and stagger
  retries.
- `request_timeout` (503): you waited 120 s (text) / 150 s (VLM) for a slot.
  Reduce concurrency or payload size.

## Models Available

**Note**: Models are loaded on-demand. Up to 3 models can be loaded simultaneously based on available RAM.

- **Know your active profile**: call `GET /api/v1/system/model-profile` to see which hardware profile was selected, which models are preloaded/warmed, and what RAM assumptions were used after auto-detection.

- **Primary**: `qwen3-vl:8b-instruct-q4_K_M` (8B parameters, quantized vision-language model) ⭐ **VLM SUPPORTED**
  - **Optimized for laptops**: Quantized Q4_K_M build keeps RAM usage around ~6 GB while retaining Qwen 3 multimodal features
  - **Full multimodal capabilities**: Images + text, OCR, chart/table understanding, spatial reasoning
  - **Dual format support**: Native Ollama format + OpenAI-compatible requests (Docling ready)
  - **128K context window**: Plenty of headroom for long document/image conversations
  - **Fast load + low power**: Smaller footprint = faster cold starts on 32 GB MacBook Pros
- **Secondary**: `qwen3:14b-q4_K_M` (14B parameters, quantized dense text model)
  - **Hybrid reasoning**: Qwen 3 “thinking vs. fast” modes for better latency control
  - **128K context window**: Handles long chat histories and RAG prompts
  - **High-quality responses**: 14B dense backbone with 36T-token training run
  - **Fits comfortably**: ~8 GB RAM when loaded—ideal default text model for this hardware
- **High-memory profile (≥ 33 GB)**: `qwen3-vl:32b` (VLM) + `qwen3:30b` (text)
  - Automatically selected when `config/models.yaml` resolves to the high-memory profile
  - Full-precision multimodal reasoning with 128K+ context and hybrid thinking
  - Ideal for workstation/desktop servers running agentic or heavy RAG workloads

Models remain in memory for 5 minutes after last use (configurable via `idle_timeout`), then are automatically unloaded to free memory by the background cleanup service. The cleanup service also monitors system memory and aggressively unloads models when memory usage exceeds 85% to prevent memory exhaustion. Switching between models requires a brief load time (~2-3 seconds).

## Vision Language Model (VLM) Support

The service **fully supports** vision-language models with **both native Ollama and OpenAI-compatible formats**:

- **Native Ollama Format** (`/api/v1/vlm`): Simple, efficient, direct integration
- **OpenAI-Compatible Format** (`/api/v1/vlm/openai`): For Docling and other OpenAI-compatible clients

**Quick Example:**

```python
import requests
import base64

with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "messages": [{"role": "user", "content": "What's in this image?"}],
        "images": [f"data:image/jpeg;base64,{img_data}"]
    }
)
print(response.json()["message"]["content"])
```

**📖 Complete VLM Guide**: See [docs/VLM_GUIDE.md](docs/VLM_GUIDE.md) for detailed examples, batch processing, streaming, and best practices.

## New Features (Latest Release)

### Embeddings API
Generate vector embeddings for semantic search, RAG systems, and similarity matching:

```bash
curl -X POST http://0.0.0.0:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:14b-q4_K_M",
    "prompt": "What is machine learning?"
  }'
```

### Model Management
Full CRUD operations for managing models:

- **List running models**: `GET /api/v1/models/ps`
- **Get model details**: `GET /api/v1/models/{name}/show`
- **Create custom models**: `POST /api/v1/models/create`
- **Copy models**: `POST /api/v1/models/{name}/copy`

### Fine-tuning Helpers
Scripts for local model customization:

```bash
# Create a Modelfile
python scripts/maintenance/fine_tune_helper.py create-modelfile \
  --base-model qwen3:14b-q4_K_M \
  --system-prompt "You are a helpful coding assistant" \
  --output Modelfile

# Create model via API
python scripts/maintenance/fine_tune_helper.py create-model \
  --name custom-assistant \
  --modelfile Modelfile
```

### Automatic Memory Management
Background service automatically:
- Unloads idle models after 5 minutes of inactivity
- Monitors system memory and unloads models when memory > 85%
- Runs cleanup checks every 60 seconds
- Prevents memory exhaustion on single-machine setups

### Agent System (Ollama 0.13.5+)
Support for Ollama's agent framework:

- **List agents**: `GET /api/v1/agents`
- **Run agent**: `POST /api/v1/agents/{name}/run`
- **Create agent**: `POST /api/v1/agents/create`

### Response Caching
Intelligent caching system with:
- Semantic similarity matching (95% threshold)
- LRU eviction policy
- Configurable TTL (default: 1 hour)
- Thread-safe operations
- Cache statistics tracking

## Documentation

Complete documentation is available in the `docs/` directory:

### Quick Start & Usage
- **[Client Guide](docs/CLIENT_GUIDE.md)** - Quick start examples for curl, Python, TypeScript, and Go
- **[VLM Guide](docs/VLM_GUIDE.md)** - Complete vision-language model guide with examples
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)** - How to integrate the service into your projects
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation

### Advanced Features
- **[POML Guide](docs/POML_GUIDE.md)** - Prompt Orchestration Markup Language support
- **[LiteLLM Guide](docs/LITELLM_GUIDE.md)** - LiteLLM integration guide
- **Embeddings API** - Generate vector embeddings for semantic search (`/api/v1/embeddings`)
- **Model Management** - Create, copy, and inspect models (`/api/v1/models/*`)
- **Fine-tuning Helpers** - Scripts for local model customization (`scripts/maintenance/fine_tune_helper.py`)
- **Agent System** - Ollama 0.13.5+ agent framework support (`/api/v1/agents/*`)
- **Response Caching** - Intelligent caching with semantic similarity matching
- **Automatic Memory Management** - Background service for model cleanup

### Operations & Maintenance
- **[Operations Guide](docs/OPERATIONS.md)** - Service operations, warm-up, and pre-loading
- **[Monitoring Guide](docs/MONITORING.md)** - Monitoring, metrics, and observability
- **[Resource Management](docs/RESOURCE_MANAGEMENT.md)** - Memory usage and performance tuning
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Configuration & Development
- **[Configuration Guide](docs/CONFIGURATION.md)** - Complete configuration reference
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture and design
- **[Development Guide](docs/DEVELOPMENT.md)** - Development setup and guidelines
- **[Stability Plan](docs/STABILITY_PLAN.md)** - Hardening roadmap

**📚 Full Documentation Index**: See [docs/README.md](docs/README.md) for the complete documentation index.

## Installation

### Quick Install

```bash
# Install Ollama (native, optimized for Apple Silicon)
./scripts/install_native.sh

# Start the service
./scripts/core/start.sh

# Verify it's running
curl http://0.0.0.0:8000/api/v1/health
```

### Pre-download Models

```bash
# Pre-download all required models
./scripts/preload_models.sh
```

**📖 Complete Installation Guide**: See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for detailed installation and configuration instructions.

## Contributing

When adding new models or modifying the service:

1. Update installation scripts if needed
2. Update preload/warmup scripts with new models
3. Update this README
4. Test with all projects
5. Document model size and use cases

**📖 Development Guide**: See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development setup, testing, and contribution guidelines.

## API Endpoints Summary

### Core Endpoints
- `POST /api/v1/generate` - Text generation
- `POST /api/v1/chat` - Chat completion (native format)
- `POST /api/v1/vlm` - Vision-language model (native format)
- `POST /api/v1/vlm/openai` - Vision-language model (OpenAI format)
- `POST /api/v1/embeddings` - Generate embeddings ⭐ NEW
- `POST /api/v1/batch/chat` - Batch chat processing
- `POST /api/v1/batch/vlm` - Batch VLM processing

### Model Management
- `GET /api/v1/models` - List all available models
- `GET /api/v1/models/ps` - List running models ⭐ NEW
- `GET /api/v1/models/{name}/show` - Get model details ⭐ NEW
- `POST /api/v1/models/create` - Create custom model ⭐ NEW
- `POST /api/v1/models/{name}/copy` - Copy model ⭐ NEW

### Agent System
- `GET /api/v1/agents` - List agents ⭐ NEW
- `POST /api/v1/agents/{name}/run` - Run agent ⭐ NEW
- `POST /api/v1/agents/create` - Create agent ⭐ NEW

### System & Monitoring
- `GET /api/v1/health` - Health check
- `GET /api/v1/queue/stats` - Queue statistics
- `GET /api/v1/metrics` - Service metrics
- `GET /api/v1/performance/stats` - Performance statistics
- `GET /api/v1/analytics` - Analytics report
- `GET /api/v1/system/model-profile` - Hardware profile and model recommendations

## License

MIT

## Support

For issues or questions:

- Check logs: `tail -f logs/api.log`
- Run health check: `./scripts/diagnostics/health_check.sh`
- Verify service: `curl http://0.0.0.0:8000/api/v1/health`
- See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions
- Open issue in project repository
