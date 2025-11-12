# Scaling & Load Testing Guide

This guide covers best practices for exercising the shared Ollama service under load and tuning the async client for high concurrency.

## Async Client Tuning

`AsyncSharedOllamaClient` exposes several configuration knobs via `AsyncOllamaConfig`:

| Setting | Default | Description |
| ------- | ------- | ----------- |
| `max_connections` | 50 | Upper bound on total HTTP connections maintained by the client |
| `max_keepalive_connections` | 20 | Number of idle keep-alive connections to retain |
| `max_concurrent_requests` | `None` | Optional semaphore to throttle in-flight requests |
| `timeout` | 300s | Per-request read timeout for generation/chat calls |
| `health_check_timeout` | 5s | Timeout used for `/api/tags` probes |
| `max_retries` / `retry_delay` | 3 / 1s | Connection verification retry strategy |

Example:

```python
from shared_ollama import AsyncOllamaConfig, AsyncSharedOllamaClient

config = AsyncOllamaConfig(
    base_url="http://ollama.internal:11434",
    max_connections=200,
    max_keepalive_connections=100,
    max_concurrent_requests=80,
    timeout=180,
)
client = AsyncSharedOllamaClient(config=config, verify_on_init=False)
```

## Locust Load Test (Async)

1. Install optional tooling (preferably in a dedicated environment):

   ```bash
   pip install locust httpx -c constraints.txt
   ```

2. Create `load_test_locust.py`:

   ```python
   from locust import HttpUser, task, between

   class OllamaUser(HttpUser):
       wait_time = between(1, 5)

       @task
       def generate(self):
           payload = {
               "model": "qwen2.5:7b",
               "prompt": "Explain streaming APIs in plain terms.",
               "stream": False,
           }
           self.client.post("/api/generate", json=payload)
   ```

3. Run Locust pointing at the shared service:

   ```bash
   locust -f load_test_locust.py --host http://localhost:11434
   ```

4. Observe metrics and adjust `max_connections` / `max_concurrent_requests` to keep latency under targets.

## Headless CLI Load Test Script

For automated sweeps or CI usage, run the included script:

```bash
python scripts/async_load_test.py \
  --requests 500 \
  --workers 40 \
  --model qwen2.5:7b \
  --prompt "Summarize the latest performance optimizations in two sentences."
```

Optional flags:

- `--duration 60` – run for 60 seconds instead of a fixed number of requests
- `--output reports/ollama_load.json` – emit JSON metrics to a custom path
- `--concurrency 80 --max-connections 120` – override client pooling limits
- `--sample-count 50` – adjust how many individual request samples are retained

The script records success rate, RPS, latency percentiles, and error breakdowns. Reports default to `logs/perf_reports/async_load_test_<timestamp>.json`.

Structured per-request events (model load times, latency, errors) are also written to `logs/requests.jsonl` whenever the sync or async clients run, making it easy to correlate load tests with individual model load durations.

## scripts/performance_report.py

The repo already includes `scripts/performance_report.py` for repeatable benchmarking. Suggested cadence:

- Before and after upgrading Ollama models
- After adjusting Async client limits
- As part of a nightly regression job

Output artifacts can be archived (e.g. in `logs/perf_reports/`) for trend analysis.

## Observability Checklist

- Enable structured logging (`logger.setLevel(logging.INFO)` in calling projects) and ship to your aggregation stack.
- Expose Prometheus metrics via `shared_ollama.telemetry.metrics` to visualize RPS, latency, and error rates.
- Alert on:
  - Sustained 5xx responses from `/api/generate`
  - Connection timeout spikes
  - Long load durations (> 30s) from model activation

## Rolling Out Changes Safely

1. Apply new async config limits in staging.
2. Run synthetic load (Locust + performance report).
3. Compare latency distribution and GPU/CPU usage.
4. Roll to production with staged rollout (if applicable).
5. Monitor dashboards for at least one evaluation window.

For additional guidance, see `docs/ARCHITECTURE.md` for system context and `README.md` for operational commands.

