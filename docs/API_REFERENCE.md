# REST API Reference

The Shared Ollama Service provides a RESTful API for language-agnostic access to Ollama models.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, no authentication is required. Rate limiting is applied per IP address.

## Rate Limits

- **Generate/Chat endpoints**: 60 requests per minute per IP
- **List Models endpoint**: 30 requests per minute per IP

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when rate limit resets

## Request Headers

Optional headers for better tracking:

- `X-Project-Name`: Project identifier (e.g., "Knowledge_Machine", "Story_Machine")
  - Used for analytics and request tracking
  - Appears in structured logs

## Endpoints

### Health Check

**GET** `/api/v1/health`

Check the health status of the API and underlying Ollama service.

**Response:**
```json
{
  "status": "healthy",
  "ollama_service": "healthy",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/health
```

---

### List Models

**GET** `/api/v1/models`

List all available models in the Ollama service.

**Response:**
```json
{
  "models": [
    {
      "name": "qwen2.5vl:7b",
      "size": 5969245856,
      "modified_at": "2025-11-11T11:33:20.071224043Z"
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/models
```

---

### Generate Text

**POST** `/api/v1/generate`

Generate text from a prompt.

**Request Body:**
```json
{
  "prompt": "Hello, world!",
  "model": "qwen2.5vl:7b",
  "system": "You are a helpful assistant.",
  "stream": false,
  "temperature": 0.2,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 100,
  "seed": 42,
  "stop": ["\n\n"]
}
```

**Parameters:**
- `prompt` (required): The prompt to generate text from
- `model` (optional): Model to use (defaults to service default)
- `system` (optional): System message for the model
- `stream` (optional): Whether to stream the response (default: false)
- `temperature` (optional): Sampling temperature (0.0-2.0)
- `top_p` (optional): Top-p sampling parameter (0.0-1.0)
- `top_k` (optional): Top-k sampling parameter
- `max_tokens` (optional): Maximum tokens to generate
- `seed` (optional): Random seed for reproducibility
- `stop` (optional): Stop sequences

**Response:**
```json
{
  "text": "Hello! How can I help you today?",
  "model": "qwen2.5vl:7b",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "latency_ms": 1234.56,
  "model_load_ms": 200.0,
  "model_warm_start": false,
  "prompt_eval_count": 16,
  "generation_eval_count": 8,
  "total_duration_ms": 500.0
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-Project-Name: MyProject" \
  -d '{
    "prompt": "Explain quantum computing",
    "model": "qwen2.5vl:7b"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "Explain quantum computing",
        "model": "qwen2.5vl:7b",
        "temperature": 0.7
    },
    headers={"X-Project-Name": "MyProject"}
)
data = response.json()
print(data["text"])
```

**TypeScript Example:**
```typescript
const response = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Project-Name": "MyProject"
  },
  body: JSON.stringify({
    prompt: "Explain quantum computing",
    model: "qwen2.5vl:7b",
    temperature: 0.7
  })
});
const data = await response.json();
console.log(data.text);
```

---

### Chat Completion

**POST** `/api/v1/chat`

Process a conversation with multiple messages.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "model": "qwen2.5vl:7b",
  "stream": false,
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 100
}
```

**Parameters:**
- `messages` (required): List of chat messages
  - Each message has `role` ("user", "assistant", or "system") and `content`
- `model` (optional): Model to use (defaults to service default)
- `stream` (optional): Whether to stream the response (default: false)
- `temperature` (optional): Sampling temperature (0.0-2.0)
- `top_p` (optional): Top-p sampling parameter (0.0-1.0)
- `top_k` (optional): Top-k sampling parameter
- `max_tokens` (optional): Maximum tokens to generate
- `seed` (optional): Random seed for reproducibility
- `stop` (optional): Stop sequences

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "The capital of France is Paris."
  },
  "model": "qwen2.5vl:7b",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "latency_ms": 1234.56,
  "model_load_ms": 0.0,
  "model_warm_start": true,
  "prompt_eval_count": 20,
  "generation_eval_count": 10,
  "total_duration_ms": 450.0
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-Project-Name: MyProject" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "model": "qwen2.5vl:7b"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "model": "qwen2.5vl:7b"
    },
    headers={"X-Project-Name": "MyProject"}
)
data = response.json()
print(data["message"]["content"])
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error message",
  "error_type": "HTTPException",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid parameters, empty prompts, invalid message roles)
- `422`: Unprocessable Entity (validation errors in request body)
- `429`: Too Many Requests (rate limit exceeded, includes `Retry-After` header)
- `500`: Internal Server Error (unexpected errors)
- `503`: Service Unavailable (Ollama service connection errors)
- `504`: Gateway Timeout (request timeout, model taking too long)

**Example Error Response:**
```json
{
  "error": "Generation failed: Connection refused",
  "error_type": "ConnectionError",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Interactive API Documentation

Visit `http://localhost:8000/api/docs` for interactive Swagger UI documentation where you can:
- Explore all endpoints
- Test API calls directly in the browser
- View request/response schemas
- See example requests

---

## Structured Logging

All API requests are automatically logged to `logs/requests.jsonl` with:
- Request ID (unique per request)
- Client IP address
- Project name (from `X-Project-Name` header)
- Model used
- Operation type
- Latency metrics
- Model load times
- Token counts
- Success/error status

**Example log entry:**
```json
{
  "event": "api_request",
  "client_type": "rest_api",
  "operation": "generate",
  "status": "success",
  "model": "qwen2.5vl:7b",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "client_ip": "127.0.0.1",
  "project_name": "MyProject",
  "latency_ms": 1234.56,
  "model_load_ms": 200.0,
  "model_warm_start": false,
  "prompt_eval_count": 16,
  "generation_eval_count": 8,
  "timestamp": "2025-11-12T14:30:00.123456+00:00"
}
```

---

## Best Practices

1. **Use Project Headers**: Always include `X-Project-Name` header for better tracking
2. **Handle Rate Limits**: Check `X-RateLimit-Remaining` header and implement backoff
3. **Error Handling**: Always check response status codes and handle errors gracefully
   - `400`: Check request parameters (empty prompts, invalid roles)
   - `422`: Fix validation errors in request body
   - `429`: Implement exponential backoff using `Retry-After` header
   - `503`: Service is down, retry after checking Ollama service status
   - `504`: Request timed out, consider using a faster model or shorter prompts
4. **Request IDs**: Use the `request_id` from responses for debugging and support
5. **Model Selection**: Specify the model explicitly if you need a specific one
6. **Warm Models**: First request to a model may be slower (cold start)
7. **Input Validation**: The API validates inputs automatically, but ensure:
   - Prompts are not empty
   - Message roles are valid (`user`, `assistant`, `system`)
   - Content length is reasonable (< 1M characters)

---

## Migration from Direct Ollama API

If you're currently calling Ollama directly at `http://localhost:11434`, migrate to the REST API:

**Before:**
```python
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen2.5vl:7b", "prompt": "Hello"}
)
```

**After:**
```python
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={"model": "qwen2.5vl:7b", "prompt": "Hello"},
    headers={"X-Project-Name": "MyProject"}
)
```

**Benefits:**
- ✅ Centralized logging and metrics
- ✅ Rate limiting protection
- ✅ Request tracking
- ✅ Better error handling
- ✅ Consistent API versioning
