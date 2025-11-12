# RESTful API Wrapper Proposal

## Problem Statement

Currently, projects connect to the Shared Ollama Service in two ways:

1. **Python projects**: Import the client library directly (requires `sys.path` manipulation)
2. **Non-Python projects**: Call Ollama's API directly (bypasses our structured logging, metrics, resilience features)

### Issues with Current Approach

- ❌ **Direct HTTP calls bypass our features**: No structured logging, metrics, or resilience patterns
- ❌ **No centralized control**: Can't implement rate limiting, authentication, or request queuing
- ❌ **Python-only library**: Requires path manipulation, not ideal for production
- ❌ **Inconsistent telemetry**: Some requests logged, others not
- ❌ **No request prioritization**: All requests treated equally

## Proposed Solution: RESTful API Wrapper

Create a FastAPI-based REST API that wraps our existing client library, providing:

### Core Features

1. **Language Agnostic**: Works with Python, TypeScript, Go, Rust, etc.
2. **Centralized Logging**: ALL requests go through structured logging
3. **Unified Metrics**: Aggregated metrics across all projects
4. **Rate Limiting**: Per-project or per-IP rate limits
5. **Request Queuing**: Priority-based request handling
6. **Backward Compatible**: Existing Python library still works

### API Design

```
POST   /api/v1/generate          # Generate text
POST   /api/v1/chat               # Chat completion
GET    /api/v1/models             # List available models
GET    /api/v1/health             # Health check
GET    /api/v1/metrics            # Service metrics (optional)
POST   /api/v1/stream/generate    # Streaming generation
POST   /api/v1/stream/chat        # Streaming chat
```

### Architecture

```
┌────────────────────┐
│  Client Projects   │  (Python, TypeScript, Go, etc.)
│  (Any Language)    │
└──────────┬─────────┘
           │ HTTP/REST
           ▼
┌──────────────────────────┐
│  FastAPI REST Wrapper    │  (New layer)
│  - Rate limiting         │
│  - Authentication        │
│  - Request queuing       │
│  - Metrics aggregation   │
└──────────┬───────────────┘
           │ Uses
           ▼
┌──────────────────────────┐
│  Shared Ollama Client     │  (Existing)
│  - Structured logging     │
│  - Resilience patterns    │
│  - Telemetry              │
└──────────┬───────────────┘
           │ HTTP
           ▼
┌──────────────────────────┐
│  Ollama Service          │
│  (Port 11434)            │
└──────────────────────────┘
```

### Benefits

1. **Universal Access**: Any language can use the service
2. **Centralized Control**: Rate limiting, auth, monitoring in one place
3. **Better Observability**: All requests logged and metered
4. **Production Ready**: Proper API with versioning, docs, etc.
5. **Backward Compatible**: Python library still works for direct integration

### Implementation Plan

**Phase 1: Basic API** (MVP)
- FastAPI wrapper with generate/chat endpoints
- Wraps existing client library
- Structured logging for all requests
- Health check endpoint

**Phase 2: Enhanced Features**
- Rate limiting (per-project or per-IP)
- Request queuing/priority
- Metrics aggregation endpoint
- API documentation (OpenAPI/Swagger)

**Phase 3: Advanced Features**
- Authentication/authorization
- Request caching/deduplication
- WebSocket support for streaming
- Multi-tenant support

### Example Usage

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={"prompt": "Hello", "model": "qwen2.5vl:7b"}
)
```

**TypeScript:**
```typescript
const res = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "Hello", model: "qwen2.5vl:7b" })
});
```

**Go:**
```go
resp, err := http.Post(
    "http://localhost:8000/api/v1/generate",
    "application/json",
    bytes.NewBuffer(jsonData),
)
```

## Recommendation

**Yes, we should implement a RESTful API wrapper.** It provides:

- Better architecture for multi-language support
- Centralized control and observability
- Production-ready API with proper versioning
- Maintains backward compatibility

The wrapper would run on a separate port (e.g., 8000) while Ollama continues on 11434, giving us a clean separation of concerns.

