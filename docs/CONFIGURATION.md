# Configuration Guide

This document describes the comprehensive configuration system for the Shared Ollama Service.

## Overview

The Shared Ollama Service uses a centralized configuration system built on `pydantic-settings` that:

- Loads configuration from environment variables and detected hardware profiles
- Auto-selects sensible defaults from `config/model_profiles.yaml`
- Provides type-safe configuration access
- Validates all configuration values
- Offers sensible defaults for all settings

## Quick Start

1. Run the detection script to view your hardware profile and recommended settings:
   ```bash
   ./scripts/generate_optimal_config.sh
   ```
2. Export any overrides directly in your shell (all variables are optional)
3. Start the service - configuration is automatically applied

## Configuration Structure

Configuration is organized into logical sections:

- **Ollama**: Ollama service connection and settings
- **API**: FastAPI server configuration
- **Queue**: Request queue settings (chat and VLM)
- **Batch**: Batch processing limits
- **Image**: Image processing settings for VLM
- **Image Cache**: Image cache configuration
- **Client**: Async HTTP client settings
- **Ollama Manager**: Ollama process management settings

## Accessing Configuration

### In Python Code

```python
from shared_ollama.core.config import settings

# Access configuration values
api_host = settings.api.host
api_port = settings.api.port
queue_max_concurrent = settings.queue.chat_max_concurrent
```

### Environment Variables

All configuration can be set via environment variables using the prefix pattern:

- `API_HOST`, `API_PORT` → `settings.api.host`, `settings.api.port`
- `QUEUE_CHAT_MAX_CONCURRENT` → `settings.queue.chat_max_concurrent`
- `BATCH_CHAT_MAX_REQUESTS` → `settings.batch.chat_max_requests`

See this guide for the complete list of available variables.

## Configuration Sections

### Ollama Configuration

Controls connection to the Ollama service:

```python
settings.ollama.host          # Service host (default: "localhost")
settings.ollama.port          # Service port (default: 11434)
settings.ollama.base_url      # Full URL (overrides host/port if set)
settings.ollama.keep_alive     # Model keep-alive duration (default: "5m")
settings.ollama.debug         # Debug logging (default: False)
settings.ollama.metal          # Metal acceleration (default: True)
settings.ollama.num_gpu        # GPU cores (-1 = all, default: -1)
```

**Environment Variables:**
- `OLLAMA_HOST`
- `OLLAMA_PORT`
- `OLLAMA_BASE_URL`
- `OLLAMA_KEEP_ALIVE`
- `OLLAMA_DEBUG`
- `OLLAMA_METAL`
- `OLLAMA_NUM_GPU`
- `OLLAMA_NUM_THREAD`
- `OLLAMA_MAX_RAM`
- `OLLAMA_NUM_PARALLEL`
- `OLLAMA_ORIGINS`

### API Configuration

FastAPI server settings:

```python
settings.api.host             # Server host (default: "0.0.0.0")
settings.api.port             # Server port (default: 8000)
settings.api.reload           # Auto-reload (default: False)
settings.api.log_level        # Log level (default: "info")
settings.api.title            # API title
settings.api.version          # API version
```

**Environment Variables:**
- `API_HOST`
- `API_PORT`
- `API_RELOAD`
- `API_LOG_LEVEL`
- `API_TITLE`
- `API_VERSION`
- `API_DOCS_URL`
- `API_OPENAPI_URL`

### Queue Configuration

Request queue settings for chat and VLM endpoints:

```python
# Chat queue
settings.queue.chat_max_concurrent      # Max concurrent (default: 6)
settings.queue.chat_max_queue_size      # Max queue depth (default: 50)
settings.queue.chat_default_timeout     # Default timeout (default: 60.0s)

# VLM queue
settings.queue.vlm_max_concurrent       # Max concurrent (default: 3)
settings.queue.vlm_max_queue_size       # Max queue depth (default: 20)
settings.queue.vlm_default_timeout      # Default timeout (default: 120.0s)
```

**Environment Variables:**
- `QUEUE_CHAT_MAX_CONCURRENT`
- `QUEUE_CHAT_MAX_QUEUE_SIZE`
- `QUEUE_CHAT_DEFAULT_TIMEOUT`
- `QUEUE_VLM_MAX_CONCURRENT`
- `QUEUE_VLM_MAX_QUEUE_SIZE`
- `QUEUE_VLM_DEFAULT_TIMEOUT`

### Batch Configuration

Batch processing limits:

```python
settings.batch.chat_max_concurrent      # Max concurrent batch chat (default: 5)
settings.batch.chat_max_requests        # Max requests per batch (default: 50)
settings.batch.vlm_max_concurrent       # Max concurrent batch VLM (default: 3)
settings.batch.vlm_max_requests        # Max requests per batch (default: 20)
```

**Environment Variables:**
- `BATCH_CHAT_MAX_CONCURRENT`
- `BATCH_CHAT_MAX_REQUESTS`
- `BATCH_VLM_MAX_CONCURRENT`
- `BATCH_VLM_MAX_REQUESTS`

### Image Processing Configuration

VLM image processing settings:

```python
settings.image.max_dimension           # Max dimension in pixels (default: 2667)
settings.image.jpeg_quality            # JPEG quality 1-100 (default: 85)
settings.image.png_compression         # PNG compression 0-9 (default: 6)
settings.image.max_size_bytes          # Max size in bytes (default: 10MB)
```

**Environment Variables:**
- `IMAGE_MAX_DIMENSION`
- `IMAGE_JPEG_QUALITY`
- `IMAGE_PNG_COMPRESSION`
- `IMAGE_MAX_SIZE_BYTES`

### Image Cache Configuration

Image cache settings:

```python
settings.image_cache.max_size          # Max cached images (default: 100)
settings.image_cache.ttl_seconds       # Cache TTL (default: 3600.0s)
```

**Environment Variables:**
- `IMAGE_CACHE_MAX_SIZE`
- `IMAGE_CACHE_TTL_SECONDS`

### Client Configuration

Async HTTP client settings:

```python
settings.client.timeout                # Request timeout (default: 300s)
settings.client.health_check_timeout   # Health check timeout (default: 5s)
settings.client.max_connections        # Max HTTP connections (default: 50)
settings.client.max_keepalive_connections  # Keep-alive connections (default: 20)
settings.client.max_concurrent_requests   # Max concurrent (default: None)
settings.client.max_retries            # Max retries (default: 3)
settings.client.retry_delay           # Retry delay (default: 1.0s)
settings.client.verbose               # Verbose logging (default: False)
```

**Environment Variables:**
- `CLIENT_TIMEOUT`
- `CLIENT_HEALTH_CHECK_TIMEOUT`
- `CLIENT_MAX_CONNECTIONS`
- `CLIENT_MAX_KEEPALIVE_CONNECTIONS`
- `CLIENT_MAX_CONCURRENT_REQUESTS`
- `CLIENT_MAX_RETRIES`
- `CLIENT_RETRY_DELAY`
- `CLIENT_VERBOSE`

### Ollama Manager Configuration

Ollama process management:

```python
settings.ollama_manager.auto_detect_optimizations  # Auto-detect (default: True)
settings.ollama_manager.wait_for_ready           # Wait for ready (default: True)
settings.ollama_manager.max_wait_time            # Max wait time (default: 30s)
settings.ollama_manager.shutdown_timeout        # Shutdown timeout (default: 10s)
```

**Environment Variables:**
- `OLLAMA_MANAGER_AUTO_DETECT_OPTIMIZATIONS`
- `OLLAMA_MANAGER_WAIT_FOR_READY`
- `OLLAMA_MANAGER_MAX_WAIT_TIME`
- `OLLAMA_MANAGER_SHUTDOWN_TIMEOUT`

## Validation

All configuration values are validated:

- **Numeric ranges**: Values must be within specified min/max bounds
- **Type checking**: Values are type-checked (int, float, bool, str)
- **Enum values**: String values must match allowed options (e.g., log levels)

Invalid values will raise `ValidationError` at startup.

## Environment Variable Precedence

Configuration is loaded in this order (later overrides earlier):

1. Default values (in code)
2. Auto-selected hardware profile (`config/model_profiles.yaml`)
3. Environment variables (system/process)

## Examples

### Development Configuration

```bash
export API_HOST=127.0.0.1
export API_PORT=8000
export API_RELOAD=true
export API_LOG_LEVEL=debug
export QUEUE_CHAT_MAX_CONCURRENT=2
export QUEUE_VLM_MAX_CONCURRENT=1
```

### Production Configuration

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export API_RELOAD=false
export API_LOG_LEVEL=info
export QUEUE_CHAT_MAX_CONCURRENT=10
export QUEUE_CHAT_MAX_QUEUE_SIZE=100
export QUEUE_VLM_MAX_CONCURRENT=5
export CLIENT_MAX_CONNECTIONS=200
export CLIENT_MAX_KEEPALIVE_CONNECTIONS=100
```

### High-Performance Configuration

```bash
export QUEUE_CHAT_MAX_CONCURRENT=20
export QUEUE_CHAT_MAX_QUEUE_SIZE=200
export QUEUE_VLM_MAX_CONCURRENT=10
export QUEUE_VLM_MAX_QUEUE_SIZE=50
export CLIENT_MAX_CONNECTIONS=500
export CLIENT_MAX_KEEPALIVE_CONNECTIONS=250
export BATCH_CHAT_MAX_CONCURRENT=10
export BATCH_CHAT_MAX_REQUESTS=100
```

## Best Practices

1. **Rely on auto-detected profiles**: Let `config/model_profiles.yaml` handle defaults whenever possible

2. **Use environment variables for overrides**: Export variables in your shell or deployment system (Docker, Kubernetes, etc.)

3. **Validate configuration early**: The service validates all configuration at startup

4. **Document custom settings**: If you override defaults, document why in your deployment notes

5. **Monitor queue metrics**: Adjust queue settings based on observed metrics (`/api/v1/queue/stats`)

6. **Test configuration changes**: Test configuration changes in a staging environment before production

## Troubleshooting

### Configuration Not Loading

- Re-run `./scripts/generate_optimal_config.sh` to confirm the detected profile
- Verify environment variable names match exactly (case-insensitive but prefix matters)
- Check for typos in variable names

### Validation Errors

- Check the error message for the specific field and constraint
- Verify values are within the allowed ranges (see comments in this guide)
- Ensure numeric values are actually numbers (not strings)

### Configuration Not Applied

- Restart the service after changing configuration
- Verify the configuration is being read: check startup logs
- Use `settings` object directly in code to verify values

## See Also

- `src/shared_ollama/core/config.py` - Configuration implementation
- API documentation at `/api/docs` - Runtime configuration values

