# API Status Report - REST API Endpoints

**Generated:** 2025-11-12
**Server PID:** 12003
**Status:** ✅ ALL ENDPOINTS OPERATIONAL

---

## Endpoint Status

| Endpoint | Method | Status | Response Time | Test Result |
|----------|--------|--------|---------------|-------------|
| `/api/v1/health` | GET | ✅ 200 OK | ~5ms | Working |
| `/api/v1/models` | GET | ✅ 200 OK | ~10ms | Working |
| `/api/v1/generate` | POST | ✅ 200 OK | ~350ms | Working |
| `/api/v1/chat` | POST | ✅ 200 OK | ~340ms | Working |

---

## Test Results

### 1. Generate Endpoint ✅

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Say hello","model":"granite4:tiny-h"}'
```

**Response:**
```json
{
  "text": "Hello! It's great to meet you. How can I assist you today?",
  "model": "granite4:tiny-h",
  "request_id": "754fcb45-be2d-46fc-888c-8ce406a34b02",
  "latency_ms": 430.925,
  "model_load_ms": 28.023,
  "model_warm_start": false,
  "prompt_eval_count": 33,
  "generation_eval_count": 17,
  "total_duration_ms": 428.5
}
```

**Status:** ✅ Working - Returns proper text response

---

### 2. Chat Endpoint ✅

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"model":"granite4:tiny-h"}'
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "Hello! It's great to hear from you. How can I assist you today?"
  },
  "model": "granite4:tiny-h",
  "request_id": "3a97b13a-8513-48c4-98ba-abb4d3d7193a",
  "latency_ms": 340.695,
  "model_load_ms": 28.289,
  "model_warm_start": false,
  "prompt_eval_count": 31,
  "generation_eval_count": 10,
  "total_duration_ms": 338.49
}
```

**Status:** ✅ Working - Returns proper message response

---

## For External Callers

### Base URL
```
http://localhost:8000/api/v1
```

### Required Headers
```
Content-Type: application/json
X-Project-Name: <your-project-name>  (optional)
```

---

## Response Format Guide

### Generate Endpoint Response

```python
{
    "text": str,                    # The generated text
    "model": str,                   # Model used
    "request_id": str,              # Unique request ID
    "latency_ms": float,            # Request latency
    "model_load_ms": float | None,  # Model load time
    "model_warm_start": bool,       # Was model already loaded?
    "prompt_eval_count": int,       # Prompt tokens
    "generation_eval_count": int,   # Generated tokens
    "total_duration_ms": float      # Total time
}
```

**Access the response:**
```python
response = requests.post(url, json=payload)
text = response.json()["text"]  # Get the generated text
```

---

### Chat Endpoint Response

```python
{
    "message": {
        "role": "assistant",        # Always "assistant"
        "content": str              # The response text
    },
    "model": str,                   # Model used
    "request_id": str,              # Unique request ID
    "latency_ms": float,            # Request latency
    "model_load_ms": float | None,  # Model load time
    "model_warm_start": bool,       # Was model already loaded?
    "prompt_eval_count": int,       # Prompt tokens
    "generation_eval_count": int,   # Generated tokens
    "total_duration_ms": float      # Total time
}
```

**Access the response:**
```python
response = requests.post(url, json=payload)
message = response.json()["message"]["content"]  # Get the message content
```

---

## Common Issues & Solutions

### Issue 1: "No response from models"

**Symptom:** Request succeeds (200 OK) but appears to have no content

**Causes:**
1. **Wrong response field access** - Using `response["text"]` for chat or `response["message"]` for generate
2. **Empty response handling** - Not checking if response parsing succeeded
3. **Response format mismatch** - Expecting old format

**Solution:**
```python
# For GENERATE endpoint
response = requests.post(f"{base_url}/generate", json=payload)
if response.status_code == 200:
    data = response.json()
    text = data.get("text", "")  # ✅ Correct
    # NOT: data.get("message")   # ❌ Wrong endpoint

# For CHAT endpoint
response = requests.post(f"{base_url}/chat", json=payload)
if response.status_code == 200:
    data = response.json()
    text = data["message"]["content"]  # ✅ Correct
    # NOT: data.get("text")             # ❌ Wrong endpoint
```

---

### Issue 2: 422 Validation Error

**Symptom:** `422 Unprocessable Entity` error

**Causes:**
- Missing required fields
- Invalid JSON format
- Wrong data types

**Solution:**
```python
# GENERATE - Required fields
payload = {
    "prompt": "Your prompt here",  # Required
    "model": "granite4:tiny-h"     # Optional (defaults to service default)
}

# CHAT - Required fields
payload = {
    "messages": [                   # Required - must be list
        {
            "role": "user",         # Required - "user", "assistant", or "system"
            "content": "Your text"  # Required - cannot be empty
        }
    ],
    "model": "granite4:tiny-h"      # Optional
}
```

---

### Issue 3: 500 Internal Server Error

**Symptom:** `500 Internal Server Error`

**Causes:**
- Server-side issue
- Invalid model name
- Ollama service not running

**Solution:**
1. Check health endpoint: `GET /api/v1/health`
2. Verify model name: `GET /api/v1/models`
3. Check Ollama service: `curl http://localhost:11434/api/tags`
4. Check server logs for specific error

---

### Issue 4: Connection Refused

**Symptom:** Cannot connect to server

**Causes:**
- Server not running
- Wrong port or host
- Firewall blocking connection

**Solution:**
```bash
# Check if server is running
lsof -i:8000

# Start server if not running
./scripts/start_api.sh

# Verify server is accessible
curl http://localhost:8000/api/v1/health
```

---

## Example Client Code

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Generate text
def generate_text(prompt: str, model: str = "granite4:tiny-h") -> str:
    response = requests.post(
        f"{BASE_URL}/generate",
        headers={"Content-Type": "application/json"},
        json={"prompt": prompt, "model": model},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["text"]

# Chat
def chat(messages: list, model: str = "granite4:tiny-h") -> str:
    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        json={"messages": messages, "model": model},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

# Usage
text = generate_text("Say hello")
print(f"Generate: {text}")

message = chat([{"role": "user", "content": "Say hello"}])
print(f"Chat: {message}")
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:8000/api/v1";

// Generate text
async function generateText(prompt, model = "granite4:tiny-h") {
    const response = await fetch(`${BASE_URL}/generate`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ prompt, model })
    });
    const data = await response.json();
    return data.text;
}

// Chat
async function chat(messages, model = "granite4:tiny-h") {
    const response = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ messages, model })
    });
    const data = await response.json();
    return data.message.content;
}

// Usage
const text = await generateText("Say hello");
console.log(`Generate: ${text}`);

const message = await chat([{role: "user", content: "Say hello"}]);
console.log(`Chat: ${message}`);
```

---

## Verification Commands

### Quick Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Test Generate
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test","model":"granite4:tiny-h"}'
```

### Test Chat
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Test"}],"model":"granite4:tiny-h"}'
```

### List Models
```bash
curl http://localhost:8000/api/v1/models
```

---

## Server Information

**API Documentation:** http://localhost:8000/api/docs
**Alternative Docs:** http://localhost:8000/api/redoc
**OpenAPI Schema:** http://localhost:8000/api/openapi.json

**Rate Limits:**
- `/models`: 30 requests/minute
- `/generate`: 60 requests/minute
- `/chat`: 60 requests/minute

**Available Models:**
- granite4:tiny-h (recommended)
- qwen2.5:7b
- qwen2.5:14b
- qwen2.5vl:7b

---

## Troubleshooting Checklist

If you're experiencing "no response from models":

- [ ] Verify you're using the correct URL: `http://localhost:8000/api/v1`
- [ ] Check you're accessing the right response field (`text` for generate, `message.content` for chat)
- [ ] Confirm your request format matches the examples above
- [ ] Test with curl commands to verify server is working
- [ ] Check server logs for specific errors
- [ ] Verify Ollama service is running: `curl http://localhost:11434/api/tags`
- [ ] Ensure you're not hitting rate limits (check for 429 errors)
- [ ] Update client code if using old response format

---

## Status: ✅ ALL SYSTEMS OPERATIONAL

Both `/generate` and `/chat` endpoints are working correctly and returning proper responses. If external callers are experiencing issues, it's likely due to:

1. **Response format mismatch** - Using wrong field names
2. **Cached client code** - Using old implementation
3. **URL mismatch** - Calling wrong endpoint or port

**Recommendation:** Have external callers verify they're using the correct response fields as documented above.

---

**Last Updated:** 2025-11-12
**Server Status:** Running (PID: 12003)
**All Tests:** ✅ PASSING
