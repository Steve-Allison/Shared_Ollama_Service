# Docling_Machine Compatibility Guide

## Overview

The Shared Ollama Service is **fully compatible** with Docling_Machine's implementation plan, but requires a configuration change from direct Ollama to the Shared Ollama Service wrapper.

## Key Compatibility Points

### ✅ OpenAI-Compatible Endpoints

The Shared Ollama Service provides the OpenAI-compatible endpoints that Docling's `ApiVlmOptions` requires:

- **Text Chat**: `POST /api/v1/chat/completions`
- **VLM Chat**: `POST /api/v1/vlm/openai`

Both endpoints follow OpenAI's `/chat/completions` specification and are compatible with Docling's `ApiVlmOptions`.

### ⚠️ Important Note

**Ollama does NOT provide `/v1/chat/completions` endpoint natively.** Ollama provides:
- `/api/chat` (native format)
- `/api/generate` (native format)

The Shared Ollama Service adds the OpenAI-compatible wrapper that Docling needs.

## Configuration for Docling_Machine

### Recommended Settings

Instead of connecting directly to Ollama at port 11434, configure Docling to use the Shared Ollama Service at port 8000:

```python
# In settings_module.py
ollama_base_url: str = "http://0.0.0.0:8000"  # Shared Ollama Service
ollama_api_format: Literal["native", "openai"] = "openai"  # Use OpenAI format
```

### ApiVlmOptions Configuration

```python
from docling.datamodel.pipeline_options import ApiVlmOptions

# For VLM (with images)
vlm_options = ApiVlmOptions(
    url=f"{settings.ollama_base_url}/api/v1/vlm/openai",
    params={"model": settings.ollama_model},
    prompt=settings.vlm_prompt,
    timeout=settings.ollama_timeout,
    concurrency=settings.ollama_concurrency,
)

# For text-only chat
chat_options = ApiVlmOptions(
    url=f"{settings.ollama_base_url}/api/v1/chat/completions",
    params={"model": settings.ollama_model},
    timeout=settings.ollama_timeout,
    concurrency=settings.ollama_concurrency,
)
```

## Health Check

Update health check logic to use Shared Ollama Service:

```python
# Instead of: GET http://localhost:11434/api/tags
# Use: GET http://localhost:8000/api/v1/health

async def check_ollama_health():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/health")
        return response.status_code == 200
```

## Benefits of Using Shared Ollama Service

1. **✅ OpenAI Compatibility**: Full OpenAI-compatible endpoints
2. **✅ Automatic Management**: Ollama managed internally (no manual setup)
3. **✅ Image Processing**: Automatic compression and optimization
4. **✅ Request Queuing**: Built-in concurrency control
5. **✅ Observability**: Metrics, logging, and monitoring
6. **✅ Background Cleanup**: Automatic model memory management
7. **✅ Rate Limiting**: Built-in protection against overload

## Migration Path

### Phase 1: Update Settings

```python
# REMOVE
fastapi_url: str
fastapi_startup_timeout: int
ollama_url: str  # Custom wrapper URL

# ADD
ollama_base_url: str = "http://0.0.0.0:8000"  # Shared Ollama Service
ollama_api_format: Literal["native", "openai"] = "openai"
```

### Phase 2: Update Provider

```python
# In ollama_provider.py
class OllamaProvider:
    def __init__(self, settings: VLMSettings):
        self._options = ApiVlmOptions(
            url=f"{settings.ollama_base_url}/api/v1/vlm/openai",
            params={"model": settings.ollama_model},
            prompt=settings.vlm_prompt,
            timeout=settings.ollama_timeout,
            concurrency=settings.ollama_concurrency,
        )
```

### Phase 3: Update Health Checks

```python
# In factory.py or health_checker.py
async def check_service_health():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/health")
        return response.status_code == 200
```

## Testing

Test the integration:

```bash
# 1. Start Shared Ollama Service
cd /path/to/Shared_Ollama_Service
./scripts/core/start.sh

# 2. Verify health
curl http://localhost:8000/api/v1/health

# 3. Test VLM endpoint
curl -X POST http://localhost:8000/api/v1/vlm/openai \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl:8b-instruct-q4_K_M",
    "messages": [{
      "role": "user",
      "content": [{"type": "text", "text": "What is in this image?"}]
    }]
  }'
```

## Compatibility Matrix

| Docling Requirement | Shared Ollama Service | Status |
|---------------------|----------------------|--------|
| OpenAI-compatible format | `/api/v1/vlm/openai` | ✅ Compatible |
| `/v1/chat/completions` | `/api/v1/chat/completions` | ✅ Compatible |
| Direct Ollama connection | Managed internally | ✅ Compatible |
| Health checks | `/api/v1/health` | ✅ Compatible |
| Model management | `/api/v1/models/*` | ✅ Compatible |

## Conclusion

The Shared Ollama Service is **fully compatible** with Docling_Machine's requirements. Simply change the base URL from `http://0.0.0.0:11434` to `http://0.0.0.0:8000` and use the `/api/v1/vlm/openai` or `/api/v1/chat/completions` endpoints.

This approach provides all the benefits of the Shared Ollama Service (automatic management, observability, queuing) while maintaining full compatibility with Docling's `ApiVlmOptions`.

