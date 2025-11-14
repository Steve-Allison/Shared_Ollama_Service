# Shared Ollama Service Architecture

This document summarizes how the shared Ollama deployment, Python clients, and supporting utilities fit together.

## High-Level Components

The codebase follows **Clean Architecture** principles with strict layer separation. See [CLEAN_ARCHITECTURE_REFACTORING.md](CLEAN_ARCHITECTURE_REFACTORING.md) for detailed architecture documentation.

- **Shared Ollama Service (Ollama daemon)** – Hosts all large language models and exposes the HTTP API on `:11434`.
- **REST API Layer** (FastAPI) - Interface Adapters
  - `shared_ollama/api/server.py` – FastAPI REST API server with async endpoints (port 8000).
  - `shared_ollama/api/models.py` – Pydantic request/response models with validation.
  - `shared_ollama/api/dependencies.py` – Dependency injection for use cases.
  - `shared_ollama/api/mappers.py` – API ↔ Domain entity mapping.
  - Fully async implementation using `AsyncSharedOllamaClient` for non-blocking I/O.
  - Rate limiting, CORS, structured logging, and request tracking.
- **Application Layer** - Use Cases & Interfaces
  - `shared_ollama/application/use_cases.py` – Business workflow orchestration (GenerateUseCase, ChatUseCase, ListModelsUseCase).
  - `shared_ollama/application/interfaces.py` – Protocol definitions for dependency inversion.
- **Domain Layer** - Business Logic
  - `shared_ollama/domain/entities.py` – Core business entities (Model, GenerationRequest, ChatCompletionRequest).
  - `shared_ollama/domain/value_objects.py` – Validated value objects (ModelName, Prompt, SystemMessage).
  - `shared_ollama/domain/exceptions.py` – Domain-specific exceptions.
- **Infrastructure Layer** - External Services
  - `shared_ollama/infrastructure/adapters.py` – Adapters implementing application interfaces.
  - `shared_ollama/client/sync.py` – Synchronous adapter with retries, structured logging, and resilience hooks.
  - `shared_ollama/client/async_client.py` – Async counterpart built on `httpx.AsyncClient`.
  - `shared_ollama/core/utils.py` – Service discovery, health checks, graceful fallback helpers.
- **Operational Modules**
  - `shared_ollama/telemetry/{metrics,performance,analytics}.py` – Metrics, tracing, and model usage analytics.
  - `shared_ollama/core/resilience.py` – Circuit-breaker style protections and self-healing routines.
  - `shared_ollama/core/queue.py` – Request queue management for graceful concurrency control.
  - `shared_ollama/core/ollama_manager.py` – Ollama process lifecycle management.
- **Tooling & Scripts**
  - `scripts/*.sh` – Startup, health checks, cleanup, and performance benchmarking.
  - `scripts/start_api.sh` – Start the REST API server (manages Ollama internally).
  - `scripts/performance_report.py` – Generates latency + throughput reports for regression detection.
- **Documentation / Tests**
  - `docs/*` – Integration, migration, and operational runbooks.
  - `tests/` – Comprehensive test suite with 33+ tests covering all endpoints, error scenarios, and edge cases.
  - `tests/helpers.py` – Reusable test utilities and fixtures.

## Request Flow

### Via REST API (Recommended)

```
┌────────────────────┐    HTTP/REST
│  Client Project     │ ────────────────────────────────────────┐
│  (Any Language)     │                                           │
└────────────────────┘                                           ▼
                                                          ┌──────────────────────┐
                                                          │  FastAPI REST API    │
                                                          │  (Port 8000, Async)  │
                                                          │  - Rate limiting     │
                                                          │  - Request tracking  │
                                                          │  - Structured logs   │
                                                          └──────────────────────┘
                                                                  │  Uses
                                                                  ▼
                                                          ┌───────────────────────────┐
                                                          │ AsyncSharedOllamaClient   │
                                                          │ (httpx.AsyncClient)       │
                                                          └───────────────────────────┘
                                                                  │  http(s) JSON
                                                                  ▼
                                                          ┌──────────────────────┐
                                                          │     Ollama API      │
                                                          │  (HTTP on :11434)   │
                                                          └──────────────────────┘
                                                                  │  GGUF model loads
                                                                  ▼
                                                          ┌──────────────────────┐
                                                          │   Model Runtime      │
                                                          │  (GPU/CPU via MPS)   │
                                                          └──────────────────────┘
```

### Via Direct Client Library (Python Only)

```
┌────────────────────┐    generate() / stream()
│  Client Project     │ ────────────────────────────────────────┐
│  (Python)           │                                           │
└────────────────────┘                                           ▼
                    configure client + ensure running       ┌───────────────────────────┐
                                                            │ shared_ollama.client.sync │
                                                            │ shared_ollama.client.async │
                                                            └───────────────────────────┘
                                                                  │  http(s) JSON
                                                                  ▼
                                                            ┌──────────────────────┐
                                                            │     Ollama API      │
                                                            │  (HTTP on :11434)   │
                                                            └──────────────────────┘
                                                                  │  GGUF model loads
                                                                  ▼
                                                            ┌──────────────────────┐
                                                            │   Model Runtime      │
                                                            │  (GPU/CPU via MPS)   │
                                                            └──────────────────────┘
```

Key checkpoints:

1. **Service discovery** – `utils.get_ollama_base_url()` or env vars provide base URL.
2. **Health verification** – `utils.ensure_service_running()` + `check_service_health()` gate requests.
3. **Request execution** – Clients integrate retries, structured logging, and optional streaming.
4. **Metrics** – `monitoring` + `analytics` modules capture timings for dashboards or log ingestion.

## Runtime Environments

| Layer        | Description                                             |
|--------------|---------------------------------------------------------|
| Dev Machines | Local `.venv`, manual `ollama serve`, scripts to manage |
| CI           | Headless checks via `scripts/ci_check.sh`, mocked tests |
| Production   | Long-running Ollama daemon, monitored via logs + metrics |

### Ollama Service Configuration

- `OLLAMA_METAL=1`, `OLLAMA_NUM_GPU=-1` for Apple Silicon acceleration.
- `OLLAMA_MAX_RAM` dynamically computed by `calculate_memory_limit.sh`.
- Keep-alive of 5 minutes ensures idle models unload to reclaim memory.

## Dependency Boundaries

- **External**: Ollama CLI, Python packages (`requests`, `httpx`, `numpy`, etc.), monitoring stack.
- **Internal**: Python modules in this repo; other projects consume only public client APIs.
- **Contracts**: API surface documented in `docs/API_REFERENCE.md`; integration validation via comprehensive test suite (33+ tests) covering all endpoints, error scenarios, and edge cases. See [TESTING_PLAN.md](TESTING_PLAN.md) for testing strategy.

## Operational Checklist

1. Start Ollama (`ollama serve` or `scripts/start.sh`).
2. Verify health (`scripts/status.sh`, `scripts/verify_setup.sh`).
3. Run clients/tests with `.venv` activated (`pip install -e ".[dev]" -c constraints.txt` recommended).
4. Monitor logs (`logs/ollama.log`, `ollama.error.log`) or forward to observability platform.
5. Use `scripts/performance_report.py` to baseline latency after model upgrades.


