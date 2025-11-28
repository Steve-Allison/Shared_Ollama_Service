# Vision Language Model (VLM) Guide

The Shared Ollama Service **fully supports** vision-language models with **both native Ollama and OpenAI-compatible formats**. Choose the format that works best for your use case:

- **Native Ollama Format** (`/api/v1/vlm`): Simple, efficient, direct integration with Ollama
- **OpenAI-Compatible Format** (`/api/v1/vlm/openai`): For Docling and other OpenAI-compatible clients

Both endpoints are optimized for `qwen3-vl:8b-instruct-q4_K_M` and share the same image processing pipeline.

## VLM Capabilities

- ✅ **Dual Format Support**: Native Ollama **and** OpenAI-compatible formats
- ✅ **Dedicated VLM Endpoints**: `/api/v1/vlm` (native) and `/api/v1/vlm/openai` (OpenAI-compatible)
- ✅ **Image Compression**: Automatic JPEG/PNG/WebP compression for faster processing
- ✅ **Image Caching**: LRU cache with 1-hour TTL for deduplicated images
- ✅ **Batch Processing**: `/api/v1/batch/vlm` for multiple VLM requests (max 20)
- ✅ **Separate Queues**: VLM queue (3 concurrent, 150s timeout) separate from text chat
- ✅ **Full Streaming Support**: Stream VLM responses with images

## Endpoints

| Endpoint | Format | Purpose | Max Concurrent | Timeout | Rate Limit |
|----------|--------|---------|----------------|---------|------------|
| `/api/v1/chat` | Native Ollama | Text-only chat | 6 | 120s | 60/min |
| `/api/v1/chat/completions` | OpenAI-compatible | Text-only chat (OpenAI format) | 6 | 120s | 60/min |
| `/api/v1/vlm` | Native Ollama | VLM with images | 3 | 150s | 30/min |
| `/api/v1/vlm/openai` | OpenAI-compatible | VLM with images (OpenAI format) | 3 | 150s | 30/min |
| `/api/v1/batch/chat` | Native Ollama | Batch text chat | 5 | 120s | 10/min |
| `/api/v1/batch/chat/completions` | OpenAI-compatible | Batch text chat (OpenAI format) | 5 | 120s | 10/min |
| `/api/v1/batch/vlm` | Native Ollama | Batch VLM | 3 | 150s | 5/min |
| `/api/v1/batch/vlm/completions` | OpenAI-compatible | Batch VLM (OpenAI format) | 3 | 150s | 5/min |

## Native Ollama Format

**Pure Ollama API - Separate Images Parameter:**

```json
{
  "model": "qwen3-vl:8b-instruct-q4_K_M",
  "messages": [
    {"role": "user", "content": "What's in this image?"}
  ],
  "images": ["data:image/jpeg;base64,/9j/4AAQ..."]
}
```

**Key Features:**

- ✅ **Text-only messages**: Simple `{"role": "user", "content": "text"}` format
- ✅ **Images as separate parameter**: Native Ollama `images` array
- ✅ **No conversion layer**: Direct pass-through to Ollama
- ✅ **Image compression**: Optional JPEG/PNG/WebP compression (default: JPEG, quality 85)
- ✅ **Image caching**: SHA-256-based LRU cache, 1-hour TTL

## Model Requirements

- **VLM Model**: Use `qwen3-vl:8b-instruct-q4_K_M` for vision tasks (images + text)
  - 8B parameters (quantized), 128K context, optimized for laptops
  - Enhanced OCR (32 languages), spatial reasoning, video/document understanding
- **Text-Only**: Use `/api/v1/chat` with `qwen3:14b-q4_K_M` (laptop tier) or `qwen3:30b` (workstation tier)
  - `qwen3:14b-q4_K_M`: 14B dense, 128K context, excellent for general text tasks
  - `qwen3:30b`: 30B MoE with hybrid thinking for deep RAG/function-calling workflows
- **Image Support**: Only `qwen3-vl:8b-instruct-q4_K_M` supports images

## Image Format

- **Format**: Base64-encoded data URLs: `data:image/<format>;base64,<data>`
- **Supported**: JPEG, PNG, WebP (any format Ollama supports)
- **Compression**: JPEG (quality 85), PNG (level 6), WebP (quality 85, method 6)
- **Size**: Recommended max 10MB per image
- **Validation**: Automatic format validation

## Using VLM with REST API

### Text-Only Chat (`/api/v1/chat`)

```python
import requests

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    json={
        "model": "qwen3:14b-q4_K_M",  # Text-only model
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(response.json()["message"]["content"])
```

### Force JSON Output (OpenAI-Compatible `response_format`)

```python
import requests

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    json={
        "messages": [{"role": "user", "content": "Return a JSON object"}],
        "response_format": {"type": "json_object"}  # Automatically forwards format="json" to Ollama
    },
)
print(response.json()["message"]["content"])  # Guaranteed JSON string
```

### Custom JSON Schema Constraint

```python
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["answer"]
}

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    json={
        "messages": [{"role": "user", "content": "Summarize this in JSON"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "summary_schema", "schema": schema}
        }
    },
)
print(response.json()["message"]["content"])
```

### VLM with Images (`/api/v1/vlm` - Native Ollama Format)

```python
import requests
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Native Ollama format: images as separate parameter
response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "messages": [
            {"role": "user", "content": "What's in this image?"}
        ],
        "images": [f"data:image/jpeg;base64,{img_data}"],
        "image_compression": True,  # Optional: compress images (default)
        "compression_format": "jpeg"  # Optional: jpeg/png/webp (default: jpeg)
    }
)
result = response.json()
print(result["message"]["content"])
print(f"Images processed: {result['images_processed']}")
```

### Multiple Images (Native Ollama)

```python
import requests
import base64

def encode_image(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"

# Native Ollama: all images in separate array
response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "messages": [
            {"role": "user", "content": "Compare these images"}
        ],
        "images": [
            encode_image("image1.jpg"),
            encode_image("image2.jpg")
        ]
    }
)
```

## OpenAI-Compatible Format

For **Docling** and other **OpenAI-compatible clients**, use `/api/v1/vlm/openai` with multimodal messages (images embedded in message content). Converted internally to native Ollama format for processing.

**OpenAI-Compatible Format - Images in Message Content:**

```json
{
  "model": "qwen3-vl:8b-instruct-q4_K_M",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."
          }
        }
      ]
    }
  ]
}
```

**Key Differences:**

- ✅ **Multimodal messages**: Images embedded in message `content` array
- ✅ **Content parts**: Each message can have multiple `text` and `image_url` parts
- ✅ **OpenAI compatibility**: Works with Docling and other OpenAI-compatible clients
- ✅ **Automatic conversion**: Internally converted to native Ollama format

### Using OpenAI-Compatible Format with Python

```python
import requests
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# OpenAI-compatible format: images in message content
response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm/openai",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    }
                ]
            }
        ],
        "image_compression": True,  # Optional: compress images (default)
        "compression_format": "jpeg"  # Optional: jpeg/png/webp (default: jpeg)
    }
)
result = response.json()
print(result["message"]["content"])
print(f"Images processed: {result['images_processed']}")
```

### Multiple Images (OpenAI-Compatible)

```python
import requests
import base64

def encode_image(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"

# OpenAI format: all images in message content parts
response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm/openai",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image_url", "image_url": {"url": encode_image("image1.jpg")}},
                    {"type": "image_url", "image_url": {"url": encode_image("image2.jpg")}}
                ]
            }
        ]
    }
)
```

### Docling Integration Example

```python
import requests
import base64
from docling import DocumentConverter  # Hypothetical Docling import

# Use Docling to extract document images
converter = DocumentConverter()
doc_images = converter.extract_images("document.pdf")

# Process with OpenAI-compatible VLM endpoint
for idx, image in enumerate(doc_images):
    # Encode image
    img_data = base64.b64encode(image).decode()

    # Call OpenAI-compatible VLM endpoint
    response = requests.post(
        "http://0.0.0.0:8000/api/v1/vlm/openai",
        json={
            "model": "qwen3-vl:8b-instruct-q4_K_M",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this document page"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                    ]
                }
            ]
        }
    )
    print(f"Page {idx}: {response.json()['message']['content']}")
```

## Which Format Should I Use?

| Use Case | Recommended Format | Endpoint |
|----------|-------------------|----------|
| **Docling integration** | OpenAI-compatible | `/api/v1/vlm/openai` |
| **OpenAI-compatible clients** | OpenAI-compatible | `/api/v1/chat/completions` or `/api/v1/vlm/openai` |
| **OpenAI SDK/libraries** | OpenAI-compatible | `/api/v1/chat/completions` or `/api/v1/vlm/openai` |
| **Direct Ollama integration** | Native Ollama | `/api/v1/chat` or `/api/v1/vlm` |
| **Simple use cases** | Native Ollama | `/api/v1/chat` or `/api/v1/vlm` |
| **Batch processing (native)** | Native Ollama | `/api/v1/batch/chat` or `/api/v1/batch/vlm` |
| **Batch processing (OpenAI)** | OpenAI-compatible | `/api/v1/batch/chat/completions` or `/api/v1/batch/vlm/completions` |

## Batch Processing

### Batch VLM Processing (Native Ollama Format)

```python
import requests
import base64

# Encode multiple images
images = []
for path in ["photo1.jpg", "photo2.jpg", "photo3.jpg"]:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        images.append(f"data:image/jpeg;base64,{data}")

# Batch VLM request (max 20 requests)
response = requests.post(
    "http://0.0.0.0:8000/api/v1/batch/vlm",
    json={
        "requests": [
            {
                "messages": [{"role": "user", "content": "Describe this"}],
                "images": [img],
                "model": "qwen3-vl:8b-instruct-q4_K_M"
            }
            for img in images
        ],
        "compression_format": "webp"  # Use WebP for all requests
    }
)

batch_result = response.json()
print(f"Processed: {batch_result['successful']}/{batch_result['total_requests']}")
for idx, result in enumerate(batch_result['results']):
    if result['success']:
        print(f"Image {idx}: {result['data']['message']['content'][:100]}...")
```

### Batch VLM Processing (OpenAI-Compatible Format)

```python
import requests
import base64

# Encode multiple images
images = []
for path in ["photo1.jpg", "photo2.jpg", "photo3.jpg"]:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        images.append(f"data:image/jpeg;base64,{data}")

# Batch OpenAI-compatible VLM request (max 20 requests)
response = requests.post(
    "http://0.0.0.0:8000/api/v1/batch/vlm/completions",
    json={
        "requests": [
            {
                "model": "qwen3-vl:8b-instruct-q4_K_M",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {"type": "image_url", "image_url": {"url": img}}
                        ]
                    }
                ]
            }
            for img in images
        ]
    }
)

batch_result = response.json()
print(f"Processed: {batch_result['successful']}/{batch_result['total_requests']}")
for idx, result in enumerate(batch_result['results']):
    if result['success']:
        print(f"Image {idx}: {result['data']['choices'][0]['message']['content'][:100]}...")
```

## Streaming Multimodal Responses

Streaming works seamlessly with multimodal requests:

```python
import requests
import base64
import json

with open("image.jpg", "rb") as f:
    image_url = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        chunk = json.loads(line[6:])
        print(chunk['chunk'], end='', flush=True)
        if chunk['done']:
            print(f"\n\nCompleted in {chunk['latency_ms']}ms")
```

## Best Practices

1. **Use Appropriate Models**: Always use `qwen3-vl:8b-instruct-q4_K_M` for vision tasks
2. **Optimize Images**: Resize large images before encoding to reduce payload size
3. **Error Handling**: Check for 422 (validation) errors for invalid image formats
4. **Streaming**: Use streaming for long responses to improve UX
5. **Rate Limits**: Be aware of 30 requests/minute rate limit for VLM endpoint
6. **Image Size**: Keep images under 10MB for best performance
7. **Native Format**: The service uses Ollama's native API format internally - no custom formats or proprietary code
8. **Project Tracking**: Include `X-Project-Name` header for project-based analytics and usage tracking

## VLM Request Tracking

VLM requests automatically track comprehensive metrics and analytics:

- **Image Cache Statistics**: Hit rate, total cache size, entry count, and evictions
- **Compression Savings**: Bytes saved per request when image compression is enabled
- **Project Analytics**: Usage tracking by project (via `X-Project-Name` header)
- **Performance Metrics**: Token generation rates, model load times, and warm start tracking
- **Image Processing**: Number of images processed, compression format used, and cache utilization

**Example Response with Tracking Data:**

```json
{
  "message": {
    "role": "assistant",
    "content": "The image shows a beautiful landscape..."
  },
  "model": "qwen3-vl:8b-instruct-q4_K_M",
  "images_processed": 2,
  "compression_savings_bytes": 1458920,
  "latency_ms": 1250.5,
  "model_load_ms": 0,
  "model_warm_start": true
}
```

All VLM metrics are automatically logged to `logs/requests.jsonl` and can be queried via the `/api/v1/analytics` and `/api/v1/performance/stats` endpoints.

## Error Handling

The API provides clear error messages for common issues:

- **422 Unprocessable Entity**: Invalid image format (not base64, missing data URL prefix, etc.)
- **400 Bad Request**: Invalid request structure
- **503 Service Unavailable**: Ollama service not running
- **504 Gateway Timeout**: Request timeout (may occur with very large images)

Example error response:

```json
{
  "error": "Image URL must start with 'data:image/'",
  "error_type": "ValidationError",
  "request_id": "abc-123-def"
}
```

## TypeScript/JavaScript Examples

### Native Ollama Format

```typescript
import fs from 'fs';

// Encode image
const imgBuffer = fs.readFileSync('photo.jpg');
const imgB64 = imgBuffer.toString('base64');

// Native Ollama format
const response = await fetch('http://0.0.0.0:8000/api/v1/vlm', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen3-vl:8b-instruct-q4_K_M',
    messages: [
      { role: 'user', content: 'What do you see?' }
    ],
    images: [`data:image/jpeg;base64,${imgB64}`]
  })
});

const data = await response.json();
console.log(data.message.content);
```

## See Also

- [Client Guide](CLIENT_GUIDE.md) - Complete API client examples
- [API Reference](API_REFERENCE.md) - Full endpoint documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - Project integration examples

