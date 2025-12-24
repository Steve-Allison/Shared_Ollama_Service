# Docling_Machine Compatibility Guide

## Overview

The Shared Ollama Service is **fully compatible** with Docling_Machine's implementation plan, but requires a configuration change from direct Ollama to the Shared Ollama Service wrapper.

## Key Compatibility Points

### ✅ OpenAI-Compatible Endpoints

The Shared Ollama Service provides the OpenAI-compatible endpoints that Docling's `ApiVlmOptions` requires:

- **Text Chat**: `POST /api/v1/chat/completions`
- **VLM Chat**: `POST /api/v1/vlm/openai`

Both endpoints follow OpenAI's `/chat/completions` specification and are compatible with Docling's `ApiVlmOptions`.

### ✅ Important Update

**Ollama DOES provide `/v1/chat/completions` endpoint natively** (since version 0.13.5+). Ollama provides:
- `/api/chat` (native format)
- `/api/generate` (native format)
- `/v1/chat/completions` (OpenAI-compatible format) ⭐ **Native feature**
- `/v1/embeddings` (OpenAI-compatible format) ⭐ **Native feature**

**Docling can use Ollama directly** at `http://localhost:11434/v1/chat/completions` for basic OpenAI compatibility.

**However**, the Shared Ollama Service adds significant value beyond basic compatibility:
- Request queuing and rate limiting
- Image processing and compression
- Observability and monitoring
- Background model cleanup
- Batch processing

## Configuration for Docling_Machine

### Option 1: Use Ollama Directly (Simplest)

Since Ollama 0.13.5+ provides OpenAI-compatible endpoints natively, Docling can connect directly:

```python
# In settings_module.py
ollama_base_url: str = "http://0.0.0.0:11434"  # Direct Ollama
ollama_api_format: Literal["native", "openai"] = "openai"  # Use OpenAI format
```

**Pros:**
- ✅ Simplest setup - no wrapper needed
- ✅ Native Ollama OpenAI compatibility
- ✅ Lower latency (no wrapper overhead)

**Cons:**
- ❌ No request queuing
- ❌ No rate limiting
- ❌ No image processing/compression
- ❌ Limited observability

### Option 2: Use Shared Ollama Service (Recommended for Production)

For production deployments with multiple users or high traffic:

```python
# In settings_module.py
ollama_base_url: str = "http://0.0.0.0:8000"  # Shared Ollama Service
ollama_api_format: Literal["native", "openai"] = "openai"  # Use OpenAI format
```

**Pros:**
- ✅ Request queuing and rate limiting
- ✅ Image processing and compression
- ✅ Comprehensive observability
- ✅ Background model cleanup
- ✅ Production-ready features

**Cons:**
- ❌ Additional service to manage
- ❌ Slightly more complex setup

### ApiVlmOptions Configuration

#### Option 1: Direct Ollama (Port 11434)

```python
from docling.datamodel.pipeline_options import ApiVlmOptions

# For VLM (with images) - Note: Ollama's native /v1/chat/completions supports images
vlm_options = ApiVlmOptions(
    url=f"{settings.ollama_base_url}/v1/chat/completions",  # Native endpoint
    params={"model": settings.ollama_model},
    prompt=settings.vlm_prompt,
    timeout=settings.ollama_timeout,
    concurrency=settings.ollama_concurrency,
)

# For text-only chat
chat_options = ApiVlmOptions(
    url=f"{settings.ollama_base_url}/v1/chat/completions",  # Native endpoint
    params={"model": settings.ollama_model},
    timeout=settings.ollama_timeout,
    concurrency=settings.ollama_concurrency,
)
```

#### Option 2: Shared Ollama Service (Port 8000)

```python
from docling.datamodel.pipeline_options import ApiVlmOptions

# For VLM (with images) - Uses wrapper with image processing
vlm_options = ApiVlmOptions(
    url=f"{settings.ollama_base_url}/api/v1/vlm/openai",  # Wrapper endpoint
    params={"model": settings.ollama_model},
    prompt=settings.vlm_prompt,
    timeout=settings.ollama_timeout,
    concurrency=settings.ollama_concurrency,
)

# For text-only chat
chat_options = ApiVlmOptions(
    url=f"{settings.ollama_base_url}/api/v1/chat/completions",  # Wrapper endpoint
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

Docling_Machine has **two options**:

### Option 1: Direct Ollama (Simplest)
- Use `http://localhost:11434/v1/chat/completions` directly
- Native Ollama OpenAI compatibility (0.13.5+)
- No wrapper needed for basic use cases

### Option 2: Shared Ollama Service (Production)
- Use `http://localhost:8000/api/v1/chat/completions` or `/api/v1/vlm/openai`
- Adds production features: queuing, rate limiting, image processing, observability
- Recommended for multi-user or production deployments

**Both options work with Docling's `ApiVlmOptions`** - choose based on your needs!

For simple use cases, Option 1 (direct Ollama) is sufficient. For production deployments requiring reliability and observability, Option 2 (Shared Ollama Service) adds significant value.

