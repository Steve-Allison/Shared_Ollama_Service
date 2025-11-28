# Operations Guide

This guide covers service operations, warm-up, pre-loading, and maintenance for the Shared Ollama Service.

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

#### Option 1: Warm-up after service start

```bash
# Start REST API
./scripts/core/start.sh

# In another terminal, warm up models (optional)
./scripts/warmup_models.sh
```

#### Option 2: Warm-up via REST API

```bash
# Warm up a specific model via REST API
curl http://0.0.0.0:8000/api/v1/generate -d '{
  "model": "qwen3:14b-q4_K_M",
  "prompt": "Hi",
  "options": {"num_predict": 1},
  "keep_alive": "30m"
}'
```

## Keep-Alive Recommendations

### High-Traffic Production

```bash
# Keep models loaded indefinitely
KEEP_ALIVE=-1 ./scripts/warmup_models.sh
```

### Development/Testing

```bash
# Keep models loaded for active session
KEEP_ALIVE=30m ./scripts/warmup_models.sh
```

### Memory-Constrained

```bash
# Shorter keep-alive, models reload as needed
KEEP_ALIVE=10m ./scripts/warmup_models.sh
```

## Service Management

### Start Service

```bash
# Start REST API (automatically manages Ollama)
./scripts/core/start.sh
```

### Stop Service

```bash
# Stop REST API and Ollama
./scripts/core/shutdown.sh
```

### Check Status

```bash
# Quick status check
./scripts/core/status.sh

# Comprehensive health check
./scripts/diagnostics/health_check.sh
```

### Verify Setup

```bash
# Verify installation and configuration
./scripts/core/verify_setup.sh
```

## Maintenance

### Clean Up

```bash
# Clean up logs and temporary files
./scripts/core/cleanup.sh
```

### Update Models

```bash
# Pull latest model versions
ollama pull qwen3:14b-q4_K_M
ollama pull qwen3-vl:8b-instruct-q4_K_M
```

### View Logs

```bash
# View REST API logs
tail -f logs/api.log

# View error logs
tail -f logs/api.error.log

# View structured request log
tail -f logs/requests.jsonl

# View performance logs
tail -f logs/performance.jsonl
```

## Performance Tuning

### Model Parallelism

Adjust the number of parallel model instances:

```bash
export OLLAMA_NUM_PARALLEL=3  # Default: 3
./scripts/core/start.sh
```

### Memory Limits

Adjust RAG reserve for your system:

```bash
export RAG_RESERVE_GB=12  # Default: 8GB
./scripts/calculate_memory_limit.sh
```

## See Also

- [Resource Management](RESOURCE_MANAGEMENT.md) - Memory and performance
- [Monitoring Guide](MONITORING.md) - Monitoring and observability
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues

