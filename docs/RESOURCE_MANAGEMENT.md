# Resource Management Guide

This guide covers memory usage, performance tuning, and resource management for the Shared Ollama Service.

## Memory Usage

### Model Memory Requirements

**Memory Usage:**

- `qwen3-vl:8b-instruct-q4_K_M`: ~6 GB RAM when loaded (laptop profile)
- `qwen3:14b-q4_K_M`: ~8 GB RAM when loaded (laptop profile)
- `qwen3-vl:32b`: ~21 GB RAM when loaded (workstation profile)
- `qwen3:30b`: ~19 GB RAM when loaded (workstation profile)
- **Laptop total**: ~14 GB when both models loaded
- **Workstation total**: ~40 GB when both models loaded
- **Up to 3 models can run simultaneously** on machines that meet the RAM requirement (configurable via `OLLAMA_NUM_PARALLEL`)

**Behavior**: Models are automatically loaded when requested and unloaded after 5 minutes of inactivity. Both models can be active at the same time if needed, reducing switching delays.

### Memory Configuration

The service uses **model-based memory calculation** (not percentage-based) to ensure adequate memory for RAG systems and other services running on the same machine:

- **Calculation**: Based on actual model requirements (largest model × parallel count + inference buffer + overhead)
- **Reserves**: Automatically reserves memory for:
  - System overhead: 8GB
  - RAG systems: 8GB (configurable via `RAG_RESERVE_GB` environment variable)
  - Safety buffer: 4GB
- **Automatic Configuration**: The service auto-detects RAM on startup and selects the matching profile defined in `config/models.yaml` (no `.env` file required)
- **Customization**: Adjust RAG reserve if needed:

  ```bash
  export RAG_RESERVE_GB=12  # For larger RAG systems
  ./scripts/calculate_memory_limit.sh
  ```

See `scripts/calculate_memory_limit.sh` for detailed memory calculation logic.

## Performance

### Without Warm-up

- **First request**: ~2-3 seconds (model loading into memory)
- **Subsequent requests**: ~100-500ms (depends on prompt length)

### With Warm-up (Models Pre-loaded)

- **First request**: ~100-500ms (model already in memory)
- **Subsequent requests**: ~100-500ms (consistently fast)

See [Operations Guide](OPERATIONS.md) for warm-up setup.

## Performance Comparison

| Setup | First Request | Subsequent Requests | Best For |
|-------|--------------|---------------------|----------|
| **No warm-up** | 2-3 seconds | 100-500ms | Development, occasional use |
| **Warm-up (30m keep-alive)** | 100-500ms | 100-500ms | Active development |
| **Warm-up (infinite keep-alive)** | 100-500ms | 100-500ms | Production, high-traffic |

## Model Access Control

Models are stored locally: `~/.ollama/models`

## Tuning Recommendations

### For High-Traffic Production

- Keep models loaded indefinitely: `KEEP_ALIVE=-1 ./scripts/warmup_models.sh`
- Monitor queue depth: `curl http://0.0.0.0:8000/api/v1/queue/stats`
- Adjust concurrency limits in `config.toml` if needed

### For Development/Testing

- Use shorter keep-alive: `KEEP_ALIVE=30m ./scripts/warmup_models.sh`
- Models reload as needed after keep-alive expires

### For Memory-Constrained Systems

- Use shorter keep-alive: `KEEP_ALIVE=10m ./scripts/warmup_models.sh`
- Reduce `OLLAMA_NUM_PARALLEL` if needed
- Consider using smaller quantized models

## See Also

- [Operations Guide](OPERATIONS.md) - Warm-up and pre-loading
- [Monitoring Guide](MONITORING.md) - Performance monitoring
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues

