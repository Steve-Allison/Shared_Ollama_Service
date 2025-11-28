# Shared Ollama Service Stabilization Plan

This plan turns the Shared Ollama Service into a predictable backend for RAG
workloads by combining service-side guardrails, scaled execution, and clear
client contracts.  Each workstream lists *outcomes*, *owner components*, and
*first implementation steps* so we can execute incrementally.

---

## 1. Enforce Guardrails at the API Edge
**Goal:** stop oversized or runaway jobs before they hit Ollama.

- **Prompt limits:** reject or auto-summarize requests above 4K tokens (matches
  current `OLLAMA_CONTEXT_LENGTH`). Add request metadata to logs/metrics.
- **Timeout policy:** drop API timeouts from 300 s to ≤120 s, stream partial
  results, and return 429/503 once queue depth exceeds thresholds.
- **Retry etiquette:** publish headers (`Retry-After`, custom error codes) and
  implement exponential backoff in official clients.
- **Components:** `src/shared_ollama/api/routes/*`, `core/queue.py`,
  `client/*`.

## 2. Queueing & Backpressure
**Goal:** ensure RAG bursts don’t overwhelm the single Ollama worker.

- **Async work queue:** finish the queue abstraction in `core/queue.py`,
  configure `max_concurrent`, `max_queue_size`, and expose `/system/queue`
  stats.
- **Admission control:** per-user quotas, early rejection for disconnected
  clients, cancellation hooks so abandoned jobs free capacity.
- **Visibility:** emit queue depth/timeout metrics to Prometheus/OpenTelemetry.

## 3. Scale the Ollama Layer
**Goal:** supply enough execution slots without degrading single-job latency.

- **Horizontal replicas:** run multiple Ollama daemons (per model/profile) and
  dispatch via a lightweight router in the API layer.
- **Model profiles:** pre-warm the two most used models per profile, persist
  selection in `config/models.yaml`, and provide `/system/profile` guidance.
- **Parallelism:** evaluate raising `OLLAMA_NUM_PARALLEL` only when RAM/GPU
  budgets allow; otherwise rely on replicas.

## 4. Prompt Hygiene & RAG Fit
**Goal:** shrink payloads and make behavior predictable.

- **Conversation trimming:** keep only relevant turns; auto-summarize older
  history server-side when possible.
- **Document chunking:** encourage callers to embed/summarize documents before
  hitting generation; provide helper utilities in `examples/` and `clients/`.
- **Streaming-first responses:** default SDKs to streaming so users see tokens
  quickly and can abort early.

## 5. Observability & Auto-Recovery
**Goal:** detect and heal saturation before users notice.

- **Metrics:** export latency, queue depth, truncation count, and timeout rate
  via `/metrics`; wire into existing dashboards.
- **Health gates:** each request pings a lightweight Ollama health endpoint; if
  unhealthy, we short-circuit with a structured 503 and trigger auto-restart.
- **Runbooks:** document “queue full”, “model stuck”, “OOM” procedures inside
  `docs/OPERATIONS.md`.

## 6. Client Contract & Documentation
**Goal:** every tenant knows the rules.

- **README updates:** new “Client Usage Guidelines” section (prompt limits,
  concurrency, retry policy, streaming expectations, troubleshooting).
- **Quick-start snippets:** include recommended `retry-after` handling and
  chunking helpers in Python/TS examples.
- **Change comms:** version the limits (e.g., `X-Service-Limits` header) so
  client teams can adapt as we tune the service.

## 7. Validation & SLOs
**Goal:** prove correctness before rollout and keep it measurable afterward.

- **Load testing:** k6/Locust scenarios that mimic RAG bursts, validating queue
  behavior and head-of-line protection.
- **SLO definition:** e.g., P95 completion <45 s, timeout rate <1%, queue
  rejection <5%; wire alerts to ops chat.
- **Canary rollout:** gate new limits/replicas behind feature flags, monitor,
  then widen once stable.

---

### Immediate Next Steps
1. Land API/README documentation (this change).
2. Implement prompt-length validation + fast-fail responses.
3. Enable queue/backpressure + metrics.
4. Provision second Ollama worker and router.

Track progress in the engineering board; each workstream should map to explicit
issues/PRs to keep scope manageable.

