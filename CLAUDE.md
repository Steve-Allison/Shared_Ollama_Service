# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture: Clean Architecture (Strict)

This project follows **Clean Architecture** with enforced dependency rules. Dependencies flow inward only:

```
External (HTTP/Files) → Interface Adapters → Application → Domain ← Infrastructure
```

**Critical Rules:**
- Domain layer (`domain/`) has ZERO external dependencies - only stdlib and typing
- Application layer (`application/`) depends ONLY on domain layer
- Infrastructure (`infrastructure/`, `client/`, `core/`) implements application interfaces
- API layer (`api/`) coordinates between HTTP and application use cases

**Violation Example (DO NOT DO):**
```python
# In domain/entities.py - WRONG
from fastapi import HTTPException  # ❌ Framework dependency in domain

# Correct approach:
from shared_ollama.domain.exceptions import InvalidRequestError  # ✅ Pure domain
```

## Layer Responsibilities

| Layer | Location | Can Import From | Cannot Import |
|-------|----------|-----------------|---------------|
| **Domain** | `domain/` | Nothing (stdlib only) | Any framework, infrastructure |
| **Application** | `application/` | `domain/` only | FastAPI, httpx, Pillow, etc. |
| **Infrastructure** | `infrastructure/`, `client/`, `core/` | All layers | N/A (implements interfaces) |
| **API** | `api/` | `application/`, `domain/` | Direct Ollama client calls |

## Key Architectural Patterns

### 1. Dependency Injection via Protocols

Application defines interfaces as `Protocol` types; infrastructure provides implementations:

```python
# application/interfaces.py
class OllamaClientInterface(Protocol):
    async def chat(self, messages: list[dict], ...) -> dict: ...

# application/use_cases.py
class ChatUseCase:
    def __init__(self, client: OllamaClientInterface):  # Protocol, not concrete
        self._client = client

# infrastructure/adapters.py
class OllamaAdapter:  # Implements protocol implicitly
    async def chat(self, messages: list[dict], ...) -> dict:
        return await self._async_client.chat(...)

# api/dependencies.py - Dependency injection
def get_chat_use_case() -> ChatUseCase:
    return ChatUseCase(client=OllamaAdapter())  # Inject concrete implementation
```

### 2. Request Flow (Typical)

```
FastAPI Route → Use Case (validates domain) → Infrastructure Adapter → Ollama HTTP API
       ↓              ↓                                ↓
  Pydantic     Domain Entities              AsyncSharedOllamaClient
   Models      (immutable)                    (httpx wrapper)
```

**Mappers** convert between layers at boundaries:
- `api/mappers.py`: API models ↔ Domain entities
- API models use Pydantic for validation
- Domain entities use frozen dataclasses with `__post_init__` validation

### 3. Tool Calling / POML Support

The service supports POML (Prompt Orchestration Markup Language) with:
- Tool/function calling via `tools` parameter
- JSON schema output validation via `format` parameter
- All endpoints support both: `/api/v1/generate`, `/api/v1/chat`, `/api/v1/vlm`

**Domain Entities:**
- `Tool`, `ToolFunction` - Tool definitions
- `ToolCall`, `ToolCallFunction` - Model-generated tool calls
- `ChatMessage.tool_calls` - Tools called by assistant
- `ChatMessage.tool_call_id` - For role="tool" responses

## Development Commands

### Essential Workflow

```bash
# Setup (first time)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -c constraints.txt
./scripts/verify_setup.sh

# Start service
./scripts/start.sh                    # Manages Ollama + REST API
# Service runs at: http://localhost:8000
# API docs at: http://localhost:8000/api/docs

# Stop service
./scripts/shutdown.sh                 # Graceful shutdown

# Check service status
./scripts/status.sh                   # Shows metrics, model cache, queues
./scripts/health_check.sh             # Quick health check
```

### Testing & Quality

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_api_server.py::test_chat_completion -v

# Coverage report
pytest --cov=shared_ollama --cov-report=html

# Pre-commit checks (run before committing)
make check                            # lint + format-check + type-check

# Auto-fix issues
make format                           # Auto-format code
make fix                              # Auto-fix linting issues

# Full development cycle
make all                              # clean → install-dev → format → fix → type-check → test
```

### CI Pre-Flight

```bash
./scripts/ci_check.sh                 # Validates service before pytest
# This is run automatically in CI before tests
```

## Code Style & Type Checking

- **Python Version:** 3.13+ (uses modern type syntax: `str | None`, generics without imports)
- **Linter:** Ruff (comprehensive rules)
- **Type Checker:** Pyright (strict mode)
- **Formatter:** Ruff (Black-compatible)
- **Docstrings:** Google-style with full type annotations

**Type Hints Example:**
```python
# Modern Python 3.13+ syntax (NO imports needed)
def process(data: dict[str, Any]) -> list[str] | None:
    """Process data and return results.

    Args:
        data: Input dictionary with string keys.

    Returns:
        List of results, or None if processing failed.
    """
    ...
```

## Request Queue Management

**Concurrent Request Limits:**
- Chat: 6 concurrent, 60s timeout
- VLM: 3 concurrent, 120s timeout
- Batch: 50 chat / 20 VLM per batch

Implementation: `core/queue.py` - `RequestQueue` with semaphore-based throttling

## Vision-Language Models (VLM)

**Two Format Support:**

1. **Native Ollama Format** (`/api/v1/vlm`):
   - Text messages separate from images
   - Images in `images` array parameter
   - Used by: internal clients

2. **OpenAI-Compatible Format** (`/api/v1/vlm/openai`):
   - Images embedded in message `content` as `image_url` parts
   - Converted to native format by mappers
   - Used by: Docling, OpenAI-compatible clients

**Image Processing:**
- Compression: JPEG (quality 85), PNG (level 6), WebP (method 6)
- Caching: LRU with SHA-256 dedup, 1-hour TTL, 1GB limit
- Location: `infrastructure/image_cache.py`, `infrastructure/image_processing.py`

## Modular Route Files

Server endpoints extracted to `api/routes/`:

```python
# api/server.py (127 lines - minimal)
from api.routes import system_router, generation_router, chat_router, vlm_router, batch_router

app.include_router(system_router, prefix="/api/v1")
app.include_router(generation_router, prefix="/api/v1")
# ...

# api/routes/chat.py (self-contained)
router = APIRouter()

@router.post("/chat")
async def chat(request: Request, use_case: ChatUseCase = Depends(get_chat_use_case)):
    # Full implementation here
```

**Pattern:** Each route file includes its own helpers (`_map_http_status_code`, `_stream_*_sse`)

## Telemetry & Observability

**Three Systems:**

1. **Metrics** (`telemetry/metrics.py`):
   - In-memory counters, latencies
   - Auto-trimming (max 10K entries)
   - Accessed via: `/api/v1/metrics`

2. **Analytics** (`telemetry/analytics.py`):
   - Usage tracking by model/operation/project
   - Dashboard: `./scripts/view_analytics.py`

3. **Performance** (`telemetry/performance.py`):
   - Latency percentiles (p50, p95, p99)
   - Model warm-start tracking
   - Report: `./scripts/performance_report.py`

## Configuration Management

Environment variables via `.env` (see `env.example`):

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Ollama Settings
OLLAMA_HOST=localhost:11434
OLLAMA_METAL=1                 # Apple Silicon GPU acceleration
OLLAMA_NUM_GPU=-1              # Use all GPU cores
OLLAMA_KEEP_ALIVE=5m           # Model unload delay
OLLAMA_MAX_RAM=24GB            # Auto-computed for your system

# Logging
LOG_LEVEL=info
STRUCTURED_LOGGING=true        # JSON log format
```

**Config Loading:** `core/config.py` uses `pydantic-settings` for validation

## Common Development Tasks

### Adding a New Endpoint

1. Create domain entity in `domain/entities.py` (if needed)
2. Add API model in `api/models.py` (Pydantic)
3. Create mapper in `api/mappers.py` (API ↔ Domain)
4. Implement use case in `application/use_cases.py`
5. Add route handler in `api/routes/*.py`
6. Write tests in `tests/test_api_server.py`

**Example commit:** See `942da39` (server modularization) or `84fba34` (POML backend)

### Testing Against Real Ollama

```bash
# Start Ollama daemon (if not running)
ollama serve

# Or use native macOS service
./scripts/install_native.sh
./scripts/setup_launchd.sh

# Preload models (first time)
./scripts/preload_models.sh      # Downloads qwen2.5:7b, qwen2.5vl:7b, etc.

# Warm up models (optional, reduces first-request latency)
./scripts/warmup_models.sh
```

### Mock Ollama for Tests

Tests use mock server (`tests/conftest.py`):
```python
# Mock runs on port 9999, returns canned responses
# No real Ollama needed for unit tests
pytest  # Uses mock automatically
```

## Project Statistics

- **Python Version:** 3.13+
- **Source Files:** 45+ modules
- **Tests:** 33+ comprehensive test cases
- **API Endpoints:** 15+ REST endpoints
- **Architecture:** Clean Architecture (4 layers)
- **Framework:** FastAPI + Uvicorn (async)

## Important File Locations

| Purpose | File |
|---------|------|
| Domain entities | `domain/entities.py` (400+ lines) |
| API models (Pydantic) | `api/models.py` (800+ lines) |
| Use cases | `application/use_cases.py`, `application/vlm_use_cases.py` |
| Route handlers | `api/routes/*.py` (5 files) |
| Mappers (layer boundaries) | `api/mappers.py` |
| Ollama client | `client/async_client.py` (1200+ lines) |
| Configuration | `core/config.py` |
| Request queue | `core/queue.py` |
| Health checks | `infrastructure/health_checker.py` |

## Git Workflow

```bash
# Check what's changed
git status --short

# Run quality checks before committing
make check

# Commit with descriptive message
git add <files>
git commit -m "feat: Add POML tool calling support"

# Push to GitHub
git push
```

**Commit Style:** Conventional commits preferred (`feat:`, `fix:`, `refactor:`, `docs:`, etc.)

## Resources

- **API Docs:** http://localhost:8000/api/docs (when service running)
- **Architecture Diagram:** `docs/ARCHITECTURE.md`
- **Clean Architecture Details:** `docs/CLEAN_ARCHITECTURE_REFACTORING.md`
- **Testing Plan:** `docs/TESTING_PLAN.md`
- **Configuration Guide:** `docs/CONFIGURATION.md`
- **Performance Analysis:** `docs/PERFORMANCE_ANALYSIS.md`
