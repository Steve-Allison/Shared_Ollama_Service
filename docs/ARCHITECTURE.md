# Shared Ollama Service Architecture

This document summarizes how the shared Ollama deployment, Python clients, and supporting utilities fit together.

## High-Level Components

- **Shared Ollama Service (Ollama daemon)** – Hosts all large language models and exposes the HTTP API on `:11434`.
- **Python Client Layer**
  - `shared_ollama/client/sync.py` – Synchronous adapter with retries, structured logging, and resilience hooks.
  - `shared_ollama/client/async_client.py` – Async counterpart built on `httpx.AsyncClient`.
  - `shared_ollama/core/utils.py` – Service discovery, health checks, graceful fallback helpers.
- **Operational Modules**
  - `shared_ollama/telemetry/{metrics,performance,analytics}.py` – Metrics, tracing, and model usage analytics.
  - `shared_ollama/core/resilience.py` – Circuit-breaker style protections and self-healing routines.
- **Tooling & Scripts**
  - `scripts/*.sh` – Startup, health checks, cleanup, and performance benchmarking.
  - `scripts/performance_report.py` – Generates latency + throughput reports for regression detection.
- **Documentation / Tests**
  - `docs/*` – Integration, migration, and operational runbooks.
  - `tests/` – Coverage for core sync + async clients, modernization compatibility, and utilities.

## Request Flow

```
┌────────────────────┐    generate() / stream()
│  Client Project     │ ────────────────────────────────────────┐
└────────────────────┘                                           │
                                                                  ▼
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
- **Contracts**: API surface documented in `docs/API_REFERENCE.md`; integration validation via `tests/test_client.py` and `tests/test_modernization.py`.

## Operational Checklist

1. Start Ollama (`ollama serve` or `scripts/start.sh`).
2. Verify health (`scripts/status.sh`, `scripts/verify_setup.sh`).
3. Run clients/tests with `.venv` activated (`pip install -e ".[dev]" -c constraints.txt` recommended).
4. Monitor logs (`logs/ollama.log`, `ollama.error.log`) or forward to observability platform.
5. Use `scripts/performance_report.py` to baseline latency after model upgrades.


