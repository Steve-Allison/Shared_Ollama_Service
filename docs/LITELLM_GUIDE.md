# LiteLLM Integration

The Shared Ollama Service **fully supports** [LiteLLM](https://github.com/BerriAI/litellm), an open-source AI gateway that provides a unified OpenAI-compatible API for 100+ LLM providers.

## What is LiteLLM?

**LiteLLM** is an AI gateway that acts as a proxy/router, allowing clients to use one unified API format while routing to different LLM backends (OpenAI, Anthropic, Google, Hugging Face, etc.). It simplifies integration by standardizing inputs and outputs across providers.

## Why Use LiteLLM with Shared Ollama Service?

- ✅ **Unified API**: Use the same client code for multiple LLM providers
- ✅ **Cost Tracking**: Monitor usage and set budgets per project/API key
- ✅ **Load Balancing**: Automatic retry and fallback mechanisms
- ✅ **Observability**: Integration with MLflow, Langfuse, Helicone, and more
- ✅ **Structured Output**: Full support for `guided_json` and `response_format` parameters

## Supported LiteLLM Parameters

The service supports all LiteLLM-specific parameters for maximum compatibility:

### 1. `guided_json` Parameter ⭐ **NEW**

LiteLLM's `guided_json` parameter is automatically converted to OpenAI's `response_format`:

```python
import requests

# LiteLLM sends guided_json (direct JSON schema)
response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat/completions",
    json={
        "model": "qwen3:14b-q4_K_M",
        "messages": [{"role": "user", "content": "Extract structured data"}],
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
    }
)
# Automatically converted to response_format internally
```

### 2. `response_format` Parameter

Standard OpenAI `response_format` is fully supported:

```python
# Basic JSON mode
response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Return JSON"}],
        "response_format": {"type": "json_object"}
    }
)

# Structured output with schema
response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Extract data"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_data",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"}
                    },
                    "required": ["summary"]
                }
            }
        }
    }
)
```

### 3. `extra_body` Parameter

Provider-specific options (currently logged but not processed):

```python
response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Hello"}],
        "extra_body": {
            "custom_option": "value",
            "provider_specific": True
        }
    }
)
```

### 4. `metadata` Parameter

Request metadata for tracking and observability (currently logged but not processed):

```python
response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat/completions",
    json={
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {
            "user_id": "user123",
            "project": "my_project",
            "session_id": "session456"
        }
    }
)
```

## Using LiteLLM Client

You can use LiteLLM's Python client directly with the Shared Ollama Service:

```python
from litellm import completion

# Configure LiteLLM to use Shared Ollama Service
response = completion(
    model="openai/qwen3:14b-q4_K_M",  # LiteLLM format: provider/model
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="http://0.0.0.0:8000/api/v1",
    api_key="not-needed",  # Service doesn't require API key
    # LiteLLM-specific parameters
    guided_json={
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    },
    metadata={"project": "my_project"},
    extra_body={"custom": "option"}
)
```

## Parameter Conversion

The service automatically handles parameter conversion:

| LiteLLM Parameter | Converted To | Status |
|------------------|--------------|--------|
| `guided_json` | `response_format.type="json_schema"` | ✅ Automatic |
| `response_format` | Direct pass-through | ✅ Supported |
| `extra_body` | Logged (not processed) | ✅ Accepted |
| `metadata` | Logged (not processed) | ✅ Accepted |

## API Endpoints Supporting LiteLLM

All OpenAI-compatible endpoints support LiteLLM parameters:

| Endpoint | LiteLLM Support | Use Case |
|----------|----------------|----------|
| `/api/v1/chat/completions` | Full support | Text chat with LiteLLM |
| `/api/v1/vlm/openai` | Full support | VLM with LiteLLM |
| `/api/v1/batch/chat/completions` | Full support | Batch processing with LiteLLM |
| `/api/v1/batch/vlm/completions` | Full support | Batch VLM with LiteLLM |

## Error Handling

The service returns OpenAI-compatible error responses for LiteLLM compatibility:

```json
{
  "error": {
    "message": "Invalid request",
    "type": "invalid_request_error",
    "code": "validation_error"
  }
}
```

## Best Practices

1. **Use `guided_json` for Structured Output**: Prefer `guided_json` when using LiteLLM clients for structured output
2. **Metadata for Tracking**: Use `metadata` parameter to track requests across your system
3. **Error Handling**: Handle errors using LiteLLM's standard error format
4. **Streaming**: Full streaming support compatible with LiteLLM's streaming format

## See Also

- [Client Guide](CLIENT_GUIDE.md) - API client examples
- [API Reference](API_REFERENCE.md) - Full endpoint documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - Project integration examples

