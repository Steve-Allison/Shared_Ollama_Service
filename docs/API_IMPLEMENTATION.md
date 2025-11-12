# REST API Implementation Summary

## Overview

A world-class RESTful API wrapper has been implemented for the Shared Ollama Service, providing language-agnostic access to Ollama models with centralized logging, metrics, and rate limiting.

## What Was Built

### 1. FastAPI REST API Server (`src/shared_ollama/api/server.py`)
- **Endpoints**:
  - `GET /api/v1/health` - Health check
  - `GET /api/v1/models` - List available models
  - `POST /api/v1/generate` - Text generation
  - `POST /api/v1/chat` - Chat completion
- **Features**:
  - Rate limiting (60 req/min for generate/chat, 30 req/min for models)
  - Structured logging for ALL requests
  - Request tracking with unique IDs
  - Project identification via `X-Project-Name` header
  - Comprehensive error handling
  - CORS support
  - Interactive API documentation (Swagger UI)

### 2. Request/Response Models (`src/shared_ollama/api/models.py`)
- Pydantic models for request validation
- Type-safe response models
- Comprehensive field documentation

### 3. Startup Script (`scripts/start_api.sh`)
- Easy API server startup
- Automatic virtual environment detection
- Configuration via environment variables
- Health checks before starting

### 4. Documentation
- **API_REFERENCE.md**: Complete API documentation with examples
- **README.md**: Updated with REST API usage instructions
- Interactive docs at `/api/docs` when server is running

## Installation

```bash
# Install new dependencies
pip install -e ".[dev]" -c constraints.txt

# Or install API dependencies directly
pip install fastapi uvicorn[standard] slowapi pydantic
```

## Usage

### Start the API Server

```bash
./scripts/start_api.sh
```

The API will be available at `http://localhost:8000`

### Access Interactive Documentation

Visit `http://localhost:8000/api/docs` for Swagger UI

### Example Requests

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={"prompt": "Hello", "model": "qwen2.5vl:7b"},
    headers={"X-Project-Name": "MyProject"}
)
print(response.json()["text"])
```

**TypeScript:**
```typescript
const res = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Project-Name": "MyProject"
  },
  body: JSON.stringify({
    prompt: "Hello",
    model: "qwen2.5vl:7b"
  })
});
const data = await res.json();
console.log(data.text);
```

## Benefits

1. **Language Agnostic**: Works with Python, TypeScript, Go, Rust, etc.
2. **Centralized Logging**: ALL requests logged to `logs/requests.jsonl`
3. **Unified Metrics**: All requests tracked in metrics system
4. **Rate Limiting**: Protects service from overload
5. **Request Tracking**: Unique request IDs for debugging
6. **Project Identification**: Track usage by project
7. **Production Ready**: Proper error handling, validation, documentation

## Architecture

```
Client Projects (Any Language)
    ↓ HTTP/REST
FastAPI REST Wrapper (Port 8000)
    ↓ Uses
Shared Ollama Client Library
    ↓ HTTP
Ollama Service (Port 11434)
```

## Next Steps

1. **Install dependencies**: `pip install -e ".[dev]" -c constraints.txt`
2. **Start API server**: `./scripts/start_api.sh`
3. **Test the API**: Visit `http://localhost:8000/api/docs`
4. **Update projects**: Migrate projects to use REST API instead of direct Ollama calls
5. **Monitor logs**: Check `logs/requests.jsonl` for all API requests

## Configuration

Environment variables:
- `API_HOST`: API server host (default: `0.0.0.0`)
- `API_PORT`: API server port (default: `8000`)
- `OLLAMA_BASE_URL`: Ollama service URL (default: `http://localhost:11434`)

## Rate Limits

- Generate/Chat: 60 requests/minute per IP
- List Models: 30 requests/minute per IP

Rate limit headers included in responses:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

