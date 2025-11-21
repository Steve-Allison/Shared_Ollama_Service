# Shared Ollama Service

**REST API** - Centralized Ollama service with VLM support (dual format), batch processing, and image compression.

**Key Features**:

- **REST API**: FastAPI-based service (port 8000) that manages Ollama internally
- **Native Ollama Format**: Simple, efficient, direct integration
- **OpenAI-Compatible Format**: For Docling and other OpenAI-compatible clients
- **Automatic Management**: REST API automatically starts and manages Ollama (no manual setup needed)

**📚 Documentation**: See [docs/README.md](docs/README.md) for complete documentation index.

## Overview

This service provides a REST API (port 8000) that manages Ollama internally and makes it accessible to all projects:

- **Architecture snapshot**: see `docs/ARCHITECTURE.md` for diagrams, request flow, and runtime environments.
- **Clean Architecture**: see `docs/CLEAN_ARCHITECTURE_REFACTORING.md` for layer structure and dependency rules.
- **Testing strategy**: see `docs/TESTING_PLAN.md` for comprehensive testing approach and reusable components.
- **Scaling playbooks**: see `docs/SCALING_AND_LOAD_TESTING.md` for concurrency tuning and load-testing guidance.
- **Knowledge Machine**
- **Course Intelligence Compiler**
- **Story Machine**
- **Docling_Machine**

## Models Available

**Note**: Models are loaded on-demand. Up to 3 models can be loaded simultaneously based on available RAM.

- **Primary**: `qwen3-vl:8b-instruct-q4_K_M` (8B parameters, quantized vision-language model) ⭐ **VLM SUPPORTED**
  - **Optimized for laptops**: Quantized Q4_K_M build keeps RAM usage around ~6 GB while retaining Qwen 3 multimodal features
  - **Full multimodal capabilities**: Images + text, OCR, chart/table understanding, spatial reasoning
  - **Dual format support**: Native Ollama format + OpenAI-compatible requests (Docling ready)
  - **128K context window**: Plenty of headroom for long document/image conversations
  - **Fast load + low power**: Smaller footprint = faster cold starts on 32 GB MacBook Pros
- **Secondary**: `qwen3:14b-q4_K_M` (14B parameters, quantized dense text model)
  - **Hybrid reasoning**: Qwen 3 “thinking vs. fast” modes for better latency control
  - **128K context window**: Handles long chat histories and RAG prompts
  - **High-quality responses**: 14B dense backbone with 36T-token training run
  - **Fits comfortably**: ~8 GB RAM when loaded—ideal default text model for this hardware
- **High-memory profile (≥ 64 GB)**: `qwen3-vl:32b` (VLM) + `qwen3:30b` (text)
  - Automatically selected when `generate_optimal_config.sh` detects ≥ 64 GB RAM
  - Full-precision multimodal reasoning with 128K+ context and hybrid thinking
  - Ideal for workstation/desktop servers running agentic or heavy RAG workloads

Models remain in memory for 5 minutes after last use (OLLAMA_KEEP_ALIVE), then are automatically unloaded to free memory. Switching between models requires a brief load time (~2-3 seconds).

## Vision Language Model (VLM) Support

The service **fully supports** vision-language models with **both native Ollama and OpenAI-compatible formats**. Choose the format that works best for your use case:

- **Native Ollama Format** (`/api/v1/vlm`): Simple, efficient, direct integration with Ollama
- **OpenAI-Compatible Format** (`/api/v1/vlm/openai`): For Docling and other OpenAI-compatible clients

Both endpoints are optimized for `qwen3-vl:8b-instruct-q4_K_M` and share the same image processing pipeline.

### VLM Capabilities

- ✅ **Dual Format Support**: Native Ollama **and** OpenAI-compatible formats
- ✅ **Dedicated VLM Endpoints**: `/api/v1/vlm` (native) and `/api/v1/vlm/openai` (OpenAI-compatible)
- ✅ **Image Compression**: Automatic JPEG/PNG/WebP compression for faster processing
- ✅ **Image Caching**: LRU cache with 1-hour TTL for deduplicated images
- ✅ **Batch Processing**: `/api/v1/batch/vlm` for multiple VLM requests (max 20)
- ✅ **Separate Queues**: VLM queue (3 concurrent, 120s timeout) separate from text chat
- ✅ **Full Streaming Support**: Stream VLM responses with images

### Endpoints

| Endpoint | Format | Purpose | Max Concurrent | Timeout | Rate Limit |
|----------|--------|---------|----------------|---------|------------|
| `/api/v1/chat` | Native Ollama | Text-only chat | 6 | 60s | 60/min |
| `/api/v1/vlm` | Native Ollama | VLM with images | 3 | 120s | 30/min |
| `/api/v1/vlm/openai` | OpenAI-compatible | VLM with images | 3 | 120s | 30/min |
| `/api/v1/batch/chat` | Native Ollama | Batch text chat | 5 | 60s | 10/min |
| `/api/v1/batch/vlm` | Native Ollama | Batch VLM | 3 | 120s | 5/min |

### Native Ollama Format

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

### Model Requirements

- **VLM Model**: Use `qwen3-vl:8b-instruct-q4_K_M` for vision tasks (images + text)
  - 8B parameters (quantized), 128K context, optimized for laptops
  - Enhanced OCR (32 languages), spatial reasoning, video/document understanding
- **Text-Only**: Use `/api/v1/chat` with `qwen3:14b-q4_K_M` (laptop tier) or `qwen3:30b` (workstation tier)
  - `qwen3:14b-q4_K_M`: 14B dense, 128K context, excellent for general text tasks
  - `qwen3:30b`: 30B MoE with hybrid thinking for deep RAG/function-calling workflows
- **Image Support**: Only `qwen3-vl:8b-instruct-q4_K_M` supports images

### Image Format

- **Format**: Base64-encoded data URLs: `data:image/<format>;base64,<data>`
- **Supported**: JPEG, PNG, WebP (any format Ollama supports)
- **Compression**: JPEG (quality 85), PNG (level 6), WebP (quality 85, method 6)
- **Size**: Recommended max 10MB per image
- **Validation**: Automatic format validation

### Using VLM with REST API

**Text-Only Chat (`/api/v1/chat`):**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "model": "qwen3:14b-q4_K_M",  # Text-only model
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(response.json()["message"]["content"])
```

**Force JSON Output (OpenAI-Compatible `response_format`):**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "messages": [{"role": "user", "content": "Return a JSON object"}],
        "response_format": {"type": "json_object"}  # Automatically forwards format="json" to Ollama
    },
)
print(response.json()["message"]["content"])  # Guaranteed JSON string
```

**Custom JSON Schema Constraint:**

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
    "http://localhost:8000/api/v1/chat",
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

**VLM with Images (`/api/v1/vlm` - Native Ollama Format):**

```python
import requests
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Native Ollama format: images as separate parameter
response = requests.post(
    "http://localhost:8000/api/v1/vlm",
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

**Multiple Images (Native Ollama):**

```python
import requests
import base64

def encode_image(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"

# Native Ollama: all images in separate array
response = requests.post(
    "http://localhost:8000/api/v1/vlm",
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

### OpenAI-Compatible Format

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

**Using OpenAI-Compatible Format with Python:**

```python
import requests
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# OpenAI-compatible format: images in message content
response = requests.post(
    "http://localhost:8000/api/v1/vlm/openai",
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

**Multiple Images (OpenAI-Compatible):**

```python
import requests
import base64

def encode_image(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"

# OpenAI format: all images in message content parts
response = requests.post(
    "http://localhost:8000/api/v1/vlm/openai",
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

**Docling Integration Example:**

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
        "http://localhost:8000/api/v1/vlm/openai",
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

**Which Format Should I Use?**

| Use Case | Recommended Format | Endpoint |
|----------|-------------------|----------|
| **Docling integration** | OpenAI-compatible | `/api/v1/vlm/openai` |
| **OpenAI-compatible clients** | OpenAI-compatible | `/api/v1/vlm/openai` |
| **Direct Ollama integration** | Native Ollama | `/api/v1/vlm` |
| **Simple use cases** | Native Ollama | `/api/v1/vlm` |
| **Batch processing** | Native Ollama | `/api/v1/batch/vlm` |

**Batch VLM Processing:**

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
    "http://localhost:8000/api/v1/batch/vlm",
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

**TypeScript/JavaScript (Native Ollama):**

```typescript
import fs from 'fs';

// Encode image
const imgBuffer = fs.readFileSync('photo.jpg');
const imgB64 = imgBuffer.toString('base64');

// Native Ollama format
const response = await fetch('http://localhost:8000/api/v1/vlm', {
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

**Batch Text Chat:**

```python
import requests

# Batch chat request (max 50 requests)
response = requests.post(
    "http://localhost:8000/api/v1/batch/chat",
    json={
        "requests": [
            {
                "messages": [{"role": "user", "content": f"Question {i}?"}],
                "model": "qwen3:14b-q4_K_M"
            }
            for i in range(10)
        ]
    }
)

batch_result = response.json()
print(f"Completed: {batch_result['successful']}/{batch_result['total_requests']}")
print(f"Total time: {batch_result['total_time_ms']:.0f}ms")
```

### Streaming Multimodal Responses

Streaming works seamlessly with multimodal requests:

```python
import requests
import base64
import json

with open("image.jpg", "rb") as f:
    image_url = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

response = requests.post(
    "http://localhost:8000/api/v1/chat",
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

### Best Practices

1. **Use Appropriate Models**: Always use `qwen3-vl:8b-instruct-q4_K_M` for vision tasks
2. **Optimize Images**: Resize large images before encoding to reduce payload size
3. **Error Handling**: Check for 422 (validation) errors for invalid image formats
4. **Streaming**: Use streaming for long responses to improve UX
5. **Rate Limits**: Be aware of 60 requests/minute rate limit for chat endpoint
6. **Image Size**: Keep images under 10MB for best performance
7. **Native Format**: The service uses Ollama's native API format internally - no custom formats or proprietary code
8. **Project Tracking**: Include `X-Project-Name` header for project-based analytics and usage tracking

### Error Handling

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

## API Client Quickstart Guide

This guide provides `curl` examples for quickly interacting with the Shared Ollama Service API. For Python, TypeScript, and Go examples, refer to the "Using VLM with REST API" and "Usage in Projects" sections.

Before you begin, ensure the API service is running:
```bash
./scripts/start.sh
```

### 1. Text-Only Chat (`/api/v1/chat`)

Send a text-only message to a language model.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen3:14b-q4_K_M",
           "messages": [
             {"role": "user", "content": "Tell me a short story about a brave knight."}
           ]
         }'
```

**Example Response (JSON):**
```json
{
  "message": {
    "role": "assistant",
    "content": "Sir Reginald, a knight known for his polka-dotted shield..."
  },
  "model": "qwen3:14b-q4_K_M",
  "created_at": "...",
  "done": true,
  "total_duration": ...,
  "load_duration": ...,
  "prompt_eval_count": ...,
  "prompt_eval_duration": ...,
  "eval_count": ...,
  "eval_duration": ...
}
```

### 2. VLM with Images (Native Ollama Format - `/api/v1/vlm`)

Send a multimodal request with text and an image using Ollama's native format. The image data is passed as a top-level `images` array (though internally it will be associated with the last user message).

**Request:**
```bash
# First, convert your image to a base64 data URL
# Example using Python:
# python -c "import base64; print('data:image/jpeg;base64,' + base64.b64encode(open('photo.jpg', 'rb').read()).decode('utf-8'))"
IMAGE_DATA_URL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAA..." # Replace with your actual base64 image data

curl -X POST http://localhost:8000/api/v1/vlm \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen3-vl:8b-instruct-q4_K_M",
           "messages": [
             {"role": "user", "content": "Describe this image in detail."}
           ],
           "images": ["'"${IMAGE_DATA_URL}"'"],
           "image_compression": true
         }'
```

**Example Response (JSON):**
```json
{
  "message": {
    "role": "assistant",
    "content": "The image depicts a vibrant landscape with a serene lake..."
  },
  "model": "qwen3-vl:8b-instruct-q4_K_M",
  "created_at": "...",
  "done": true,
  "images_processed": 1,
  "compression_savings_bytes": ...,
  "total_duration": ...,
  "load_duration": ...,
  "prompt_eval_count": ...,
  "prompt_eval_duration": ...,
  "eval_count": ...,
  "eval_duration": ...
}
```

### 3. VLM with Images (OpenAI-Compatible Format - `/api/v1/vlm/openai`)

Send a multimodal request with text and an image using an OpenAI-compatible message format. Images are embedded directly within the message content.

**Request:**
```bash
# First, convert your image to a base64 data URL
# Example using Python:
# python -c "import base64; print('data:image/jpeg;base64,' + base64.b64encode(open('photo.jpg', 'rb').read()).decode('utf-8'))"
IMAGE_DATA_URL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAA..." # Replace with your actual base64 image data

curl -X POST http://localhost:8000/api/v1/vlm/openai \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen3-vl:8b-instruct-q4_K_M",
           "messages": [
             {
               "role": "user",
               "content": [
                 {"type": "text", "text": "What do you see in this picture?"},
                 {"type": "image_url", "image_url": {"url": "'"${IMAGE_DATA_URL}"'"}}
               ]
             }
           ],
           "image_compression": true
         }'
```

**Example Response (JSON, OpenAI-like):**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": ...,
  "model": "qwen3-vl:8b-instruct-q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image shows a bustling city street at night..."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": ...,
    "completion_tokens": ...,
    "total_tokens": ...
  }
}
```

---

## POML Support (Prompt Orchestration Markup Language)

The Shared Ollama Service **fully supports** [POML](https://github.com/microsoft/poml) for structured prompt engineering and LLM orchestration.

### What is POML?

**POML** (Prompt Orchestration Markup Language) by Microsoft is an XML-based markup language that brings structure and maintainability to LLM prompts. It's like HTML for AI prompts.

### Why Use POML?

- ✅ **Structured Prompts**: Organize complex prompts with clear structure
- ✅ **Tool/Function Calling**: Define tools with `<tool-definition>` tags
- ✅ **Template Variables**: Use `{{ variable }}` for dynamic content
- ✅ **File Inclusion**: Load content with `<document src="file.txt">`
- ✅ **Control Flow**: Use `for` and `if` attributes for logic
- ✅ **JSON Schema Output**: Structured output validation
- ✅ **Runtime Parameters**: Set model params in POML templates

### Supported POML Features

#### 1. Tool/Function Calling ⭐ **NEW**

Define functions the model can call using POML's `<tool-definition>` syntax:

**POML Template** (`chat_with_tools.poml`):

```xml
<poml>
  <system-msg>You are a helpful assistant with access to tools.</system-msg>
  <human-msg>{{ user_question }}</human-msg>

  <!-- Define available tools -->
  <tool-definition name="get_weather" description="Get current weather for a location">
  {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or location"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit"
      }
    },
    "required": ["location"]
  }
  </tool-definition>

  <tool-definition name="calculate" description="Perform mathematical calculations">
  {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "Mathematical expression to evaluate"
      }
    },
    "required": ["expression"]
  }
  </tool-definition>

  <!-- Handle tool interactions -->
  <tool-request if="tool_request" id="{{ tool_request.id }}"
                name="{{ tool_request.name }}"
                parameters="{{ tool_request.parameters }}" />
  <tool-response if="tool_response" id="{{ tool_response.id }}"
                 name="{{ tool_response.name }}">
    <object data="{{ tool_response.result }}"/>
  </tool-response>

  <runtime model="qwen3:14b-q4_K_M" temperature="0.7"/>
</poml>
```

**Python Integration**:

```python
import poml
import requests
import json

# Step 1: Generate initial request with POML
context = {"user_question": "What's the weather in Paris?"}
params = poml.poml("chat_with_tools.poml", context=context, format="openai_chat")

# Step 2: Send to Shared Ollama Service
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json=params
).json()

# Step 3: Check if model wants to call a tool
if response["message"].get("tool_calls"):
    tool_call = response["message"]["tool_calls"][0]

    # Step 4: Execute the function
    if tool_call["function"]["name"] == "get_weather":
        args = json.loads(tool_call["function"]["arguments"])
        result = {"temperature": 22, "condition": "sunny"}  # Your function

        # Step 5: Send tool result back
        context["tool_response"] = {
            "id": tool_call["id"],
            "name": tool_call["function"]["name"],
            "result": result
        }
        params = poml.poml("chat_with_tools.poml", context=context, format="openai_chat")
        final_response = requests.post(
            "http://localhost:8000/api/v1/chat",
            json=params
        ).json()

        print(final_response["message"]["content"])
        # "The weather in Paris is currently 22°C and sunny."
```

#### 2. JSON Schema Output Validation

Use POML's `<output-schema>` to ensure structured responses:

**POML Template** (`extract_event.poml`):

```xml
<poml>
  <system-msg>Extract event information from the text.</system-msg>
  <human-msg>{{ text }}</human-msg>

  <output-schema>
  {
    "type": "object",
    "properties": {
      "name": { "type": "string" },
      "date": { "type": "string" },
      "participants": { "type": "array", "items": { "type": "string" } },
      "location": { "type": "string" }
    },
    "required": ["name", "date", "participants"]
  }
  </output-schema>

  <runtime model="qwen3:14b-q4_K_M"/>
</poml>
```

**Python Integration**:

```python
import poml
import requests
import json

context = {
    "text": "Alice and Bob are meeting at the Science Fair on Friday in Building A."
}
params = poml.poml("extract_event.poml", context=context, format="openai_chat")

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json=params
).json()

event = json.loads(response["message"]["content"])
print(event)
# {
#   "name": "Science Fair",
#   "date": "Friday",
#   "participants": ["Alice", "Bob"],
#   "location": "Building A"
# }
```

#### 3. VLM with POML

Combine vision capabilities with structured prompts:

**POML Template** (`analyze_image.poml`):

```xml
<poml>
  <task>Analyze the image and extract structured information.</task>
  <human-msg>{{ question }}</human-msg>

  <output-schema>
  {
    "type": "object",
    "properties": {
      "objects": { "type": "array", "items": { "type": "string" } },
      "text_detected": { "type": "string" },
      "dominant_colors": { "type": "array", "items": { "type": "string" } },
      "scene_type": { "type": "string" }
    },
    "required": ["objects", "scene_type"]
  }
  </output-schema>

  <runtime model="qwen3-vl:8b-instruct-q4_K_M" max-tokens="500"/>
</poml>
```

**Python Integration**:

```python
import poml
import requests
import base64
import json

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Generate POML parameters
context = {"question": "What do you see in this image?"}
params = poml.poml("analyze_image.poml", context=context, format="openai_chat")

# Add images for VLM endpoint (native Ollama format)
request_data = {
    **params,
    "images": [f"data:image/jpeg;base64,{img_data}"]
}

response = requests.post(
    "http://localhost:8000/api/v1/vlm",
    json=request_data
).json()

analysis = json.loads(response["message"]["content"])
print(f"Objects: {analysis['objects']}")
print(f"Scene: {analysis['scene_type']}")
```

#### 4. Runtime Parameters

Set model parameters directly in POML:

```xml
<poml>
  <system-msg>You are a creative writer.</system-msg>
  <human-msg>{{ prompt }}</human-msg>

  <!-- Runtime parameters are automatically converted -->
  <runtime
    model="qwen3:14b-q4_K_M"
    temperature="0.9"
    max-tokens="1000"
    top-p="0.95"
    frequency-penalty="0.5"
  />
</poml>
```

### API Endpoints Supporting POML

All generation, chat, and VLM endpoints support POML-generated requests:

| Endpoint | POML Features Supported | Use Case |
|----------|------------------------|----------|
| `/api/v1/generate` | Tools, JSON schema, runtime params | Text generation with function calling |
| `/api/v1/chat` | Tools, JSON schema, runtime params | Text chat with function calling |
| `/api/v1/vlm` | Tools, JSON schema, runtime params, images | Vision + text with tools |
| `/api/v1/vlm/openai` | Tools, JSON schema, runtime params, images | OpenAI-compatible VLM |
| `/api/v1/batch/chat` | All chat features | Batch text processing |
| `/api/v1/batch/vlm` | All VLM features | Batch vision processing |

### Request Format with Tools

When POML generates requests with tools, the format is:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }
  ],
  "model": "qwen3:14b-q4_K_M",
  "temperature": 0.7
}
```

### Response Format with Tool Calls

When the model calls a tool, the response includes:

```json
{
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Paris\"}"
        }
      }
    ]
  },
  "model": "qwen3:14b-q4_K_M",
  "request_id": "req-xyz",
  "latency_ms": 450.2
}
```

### Installing POML

```bash
# Install POML Python SDK
pip install poml

# Or with specific integrations
pip install poml[agent]     # AgentOps tracing
pip install poml[mlflow]    # MLflow integration
pip install poml[weave]     # Weights & Biases Weave
```

### POML Resources

- **Official Repository**: <https://github.com/microsoft/poml>
- **Documentation**: <https://github.com/microsoft/poml/tree/main/docs>
- **Examples**: See `examples/poml/` directory in this repository
- **VS Code Extension**: Available in marketplace

### Benefits for Your Workflow

1. **Maintainable Prompts**: Store prompts as `.poml` files, version control them
2. **Reusable Components**: `<include src="common.poml" />` for shared logic
3. **Type-Safe Tools**: JSON schema validation prevents errors
4. **Template Variables**: Dynamic prompts without string concatenation
5. **Documentation**: Self-documenting prompt structure
6. **Testing**: Easier to test prompts with different contexts

### Complete Working Example

See `examples/poml/weather_assistant.py` for a full working example with:

- Multiple tool definitions
- Conversation history
- Tool call handling
- Structured output
- VLM integration

## Project Structure

The codebase follows **Clean Architecture** principles with clear separation of concerns:

```text
shared_ollama_service/
├── src/shared_ollama/
│   ├── __init__.py                 # Public SDK exports
│   ├── domain/                     # Domain layer (business logic)
│   │   ├── entities.py            # Core business entities
│   │   ├── value_objects.py        # Validated value objects
│   │   └── exceptions.py           # Domain-specific exceptions
│   ├── application/                # Application layer (use cases)
│   │   ├── interfaces.py          # Protocol definitions
│   │   └── use_cases.py            # Business workflow orchestration
│   ├── infrastructure/             # Infrastructure layer (external services)
│   │   └── adapters.py            # Adapters for external services
│   ├── api/                        # Interface adapters (HTTP layer)
│   │   ├── server.py              # FastAPI endpoints
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── dependencies.py        # Dependency injection
│   │   └── mappers.py             # API ↔ Domain mapping
│   ├── client/                     # Client implementations
│   │   ├── sync.py                # Synchronous client
│   │   └── async_client.py        # Async client (httpx-based)
│   ├── core/                       # Core utilities
│   │   ├── utils.py               # Service discovery & helpers
│   │   ├── queue.py               # Request queue management
│   │   ├── resilience.py          # Circuit breaker, retries
│   │   └── ollama_manager.py      # Ollama process management
│   └── telemetry/                  # Observability
│       ├── metrics.py             # Request telemetry & Prometheus helpers
│       ├── analytics.py           # Project-level analytics & exports
│       ├── performance.py         # Structured performance logging
│       └── structured_logging.py  # JSONL request logging
├── tests/                          # Comprehensive test suite (33+ tests)
│   ├── test_api_server.py         # API endpoint tests
│   ├── test_client.py             # Client tests
│   ├── test_async_client.py      # Async client tests
│   ├── test_resilience.py         # Resilience pattern tests
│   ├── test_telemetry.py          # Telemetry tests
│   ├── test_utils.py              # Utility tests
│   └── helpers.py                 # Reusable test utilities
└── scripts/                        # Operational CLI utilities
```

**Architecture Benefits:**

- ✅ **Clean Architecture**: Strict dependency rules (Domain → Application → Infrastructure)
- ✅ **Dependency Injection**: No global state, fully testable
- ✅ **Type Safety**: Full type hints with Python 3.13+ features
- ✅ **Testability**: 33+ comprehensive tests with reusable fixtures
- ✅ **Maintainability**: Clear separation of concerns

See [docs/CLEAN_ARCHITECTURE_REFACTORING.md](docs/CLEAN_ARCHITECTURE_REFACTORING.md) for detailed architecture documentation.

All modules are shipped as a package (installable via `pip install -e .`) with type stubs co-located under `src/shared_ollama/**/*.pyi`.

## Installation

### ⚡ Native Installation (Apple Silicon MPS Optimized)

**Best Performance**: Native Ollama with explicit MPS (Metal Performance Shaders) configuration provides maximum GPU acceleration on Apple Silicon.

```bash
# Install Ollama
./scripts/install_native.sh

# Start the REST API (automatically manages Ollama internally)
./scripts/start.sh
```

**MPS/Metal Optimizations Enabled:**

- ✅ `OLLAMA_METAL=1` - Explicit Metal/MPS GPU acceleration
- ✅ `OLLAMA_NUM_GPU=-1` - All Metal GPU cores utilized
- ✅ Maximum GPU utilization for fastest inference
- ✅ All 10 CPU cores automatically utilized
- ✅ Lower memory overhead
- ✅ Native macOS integration

### Pull Models

#### Option 1: Manual Pull

```bash
# Pull primary VLM model (qwen3-vl:8b-instruct-q4_K_M)
ollama pull qwen3-vl:8b-instruct-q4_K_M

# Pull secondary text model (qwen3:14b-q4_K_M)
ollama pull qwen3:14b-q4_K_M

```

#### Option 2: Automated Pre-download (Recommended)

```bash
# Pre-download all required models
./scripts/preload_models.sh
```

This automatically checks and downloads all required models, ensuring they're ready before first use.

### Python Development Environment

**Recommended**: Use a virtual environment for development and testing.

```bash
# Create virtual environment (modern best practice: .venv)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies (choose one method):
pip install -r requirements.txt -c constraints.txt
# OR (modern approach, preferred during development):
pip install -e .[dev] -c constraints.txt

# Deactivate when done
deactivate
```

**Dependency Management:**

- `requirements.txt` - Simple dependency list (pip install -r requirements.txt)
- `pyproject.toml` - Modern Python packaging standard (pip install -e .)
- Both files are provided for compatibility

**Install Development Tools:**

```bash
# Install with development dependencies (Ruff, Pyright, pytest)
pip install -e ".[dev]" -c constraints.txt
```

**Modern Development Tools:**

- **Ruff** (v0.14.4+) - Fast linter and formatter (replaces Black, isort, flake8, etc.)
- **Pyright** (v1.1.407+) - Type checker (Microsoft's static type checker)
- **pytest** (v9.0.0+) - Modern testing framework

**Quick Commands:**

**REST API:**

```bash
./scripts/start.sh              # Start REST API server (port 8000)
                                 # Automatically runs verify_setup.sh to check/generate optimal config
./scripts/start.sh --skip-verify # Skip verification (faster startup)
curl http://localhost:8000/api/v1/health  # Health check

# Core Endpoints
curl http://localhost:8000/api/v1/models  # List available models
curl http://localhost:8000/api/v1/chat    # Text-only chat (native Ollama)
curl http://localhost:8000/api/v1/vlm     # VLM with images (native Ollama)
curl http://localhost:8000/api/v1/vlm/openai  # VLM with images (OpenAI-compatible, for Docling)

# Monitoring & Analytics Endpoints
curl http://localhost:8000/api/v1/metrics  # Service metrics
curl http://localhost:8000/api/v1/performance/stats  # Performance statistics
curl http://localhost:8000/api/v1/analytics  # Analytics report
curl "http://localhost:8000/api/v1/analytics?project=Docling_Machine"  # Project-specific analytics
curl http://localhost:8000/api/v1/queue/stats  # Queue statistics

# Visit http://localhost:8000/api/docs for interactive API docs
# Fully async implementation for maximum concurrency
```

**Ollama Service:**

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking
pyright src/shared_ollama

# Run tests
pytest

# Run API server tests
pytest tests/test_api_server.py -v

# Run async load test script (headless)
python scripts/async_load_test.py --requests 200 --workers 20

# Or use Makefile
make lint        # Run linter
make format      # Format code
make type-check  # Type checking
make test        # Run tests
make check       # Run all checks
make fix         # Auto-fix issues
```

**Why `.venv` instead of `venv`?**

- `.venv` is the modern convention (hidden directory keeps project cleaner)
- Many modern tools (VS Code, PyCharm) auto-detect `.venv` by default
- Both are acceptable, but `.venv` is increasingly preferred in 2024+

**Note for Consuming Projects**: Projects that import this library should use their own virtual environments. The shared library can be imported via `sys.path` without needing to install its dependencies globally.

### Verify Installation

```bash
# Quick status check (recommended)
./scripts/status.sh

# Comprehensive health check
./scripts/health_check.sh

# Or check REST API health
curl http://localhost:8000/api/v1/health

# List available models (via REST API)
curl http://localhost:8000/api/v1/models
```

## Quick Start - Using in Your Projects

The easiest way to use the shared service from any project:

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")

from shared_ollama import SharedOllamaClient
from utils import ensure_service_running

# Ensure service is running (with helpful error messages)
ensure_service_running()

# Create client (auto-discovers URL from environment)
client = SharedOllamaClient()

# Use it!
response = client.generate("Hello, world!")
print(response.text)
```

**Environment Variables** (optional - defaults work too):

```bash
# For REST API (recommended)
export API_BASE_URL="http://localhost:8000"

# For direct Ollama access (not recommended - use REST API instead)
export OLLAMA_BASE_URL="http://localhost:11434"
```

**API Documentation**: See [API_REFERENCE.md](docs/API_REFERENCE.md) for complete REST API documentation.

See [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for complete integration instructions and project-specific examples.

## Usage in Projects

### 🚀 Recommended: REST API (Language Agnostic)

**Best for**: All projects (Python, TypeScript, Go, Rust, etc.)

The REST API provides centralized logging, metrics, and rate limiting for all projects. Built with FastAPI and fully async for maximum concurrency and performance.

```bash
# Start the API server (runs on port 8000 by default)
./scripts/start.sh
```

**Key Features:**

- ✅ **Fully Async**: Uses `AsyncSharedOllamaClient` for non-blocking I/O operations
- ✅ **High Concurrency**: Handles multiple concurrent requests efficiently
- ✅ **Language Agnostic**: Works with any language that supports HTTP
- ✅ **Centralized Logging**: All requests logged to structured JSON logs
- ✅ **Rate Limiting**: Protects service from overload (60 req/min for generate/chat)
- ✅ **Request Tracking**: Unique request IDs for debugging
- ✅ **Project Identification**: Track usage by project via `X-Project-Name` header
- ✅ **Input Validation**: Automatic validation of prompts, messages, and parameters
- ✅ **Enhanced Error Handling**: Specific HTTP status codes (400, 422, 429, 500, 503, 504) with actionable error messages
- ✅ **Comprehensive Testing**: Full test suite with 33+ test cases covering all endpoints, error scenarios, and edge cases
- ✅ **Clean Architecture**: Strict separation of concerns with domain, application, and infrastructure layers
- ✅ **Dependency Injection**: Fully testable with no global state, using FastAPI's dependency system

**Python Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "Hello, world!",
        "model": "qwen3:14b-q4_K_M"
    }
)
print(response.json()["text"])
```

**TypeScript/JavaScript Example:**

```typescript
const res = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Hello, world!",
    model: "qwen3:14b-q4_K_M"
  })
});
const data = await res.json();
console.log(data.text);
```

**Go Example:**

```go
resp, err := http.Post(
    "http://localhost:8000/api/v1/generate",
    "application/json",
    bytes.NewBuffer(jsonData),
)
```

**API Documentation**: Visit `http://localhost:8000/api/docs` for interactive API documentation.

**Benefits:**

- ✅ **Fully Async**: Non-blocking I/O operations for maximum concurrency
- ✅ **Language Agnostic**: Works with any language (Python, TypeScript, Go, Rust, etc.)
- ✅ **Streaming Support**: Real-time token-by-token responses for better UX
- ✅ **Request Queue**: Graceful handling of traffic spikes (up to 50 queued requests)
- ✅ **Centralized Logging**: All requests logged to structured JSON logs (`logs/requests.jsonl`)
- ✅ **Unified Metrics**: Aggregated metrics across all projects
- ✅ **Rate Limiting**: 60 requests/minute per IP for generate/chat, 30/min for models
- ✅ **Request Tracking**: Unique request IDs for debugging and support
- ✅ **Project Identification**: Track usage by project via `X-Project-Name` header
- ✅ **High Performance**: Async implementation handles concurrent requests efficiently
- ✅ **Input Validation**: Automatic validation prevents invalid requests (empty prompts, invalid roles, length limits)
- ✅ **Robust Error Handling**: Specific error responses for validation errors (422), rate limits (429), service unavailable (503), timeouts (504), and more
- ✅ **Production Ready**: Comprehensive error handling, input validation, and test coverage

### Streaming Responses

The REST API supports real-time streaming for token-by-token responses, providing better user experience for long-running AI generations.

**Python Client Streaming:**

```python
import asyncio
from shared_ollama import AsyncSharedOllamaClient

async def stream_example():
    async with AsyncSharedOllamaClient() as client:
        # Stream text generation
        async for chunk in client.generate_stream(prompt="Write a story about AI"):
            print(chunk["chunk"], end="", flush=True)
            if chunk["done"]:
                print(f"\n\nCompleted in {chunk['latency_ms']}ms")

        # Stream chat completion
        messages = [{"role": "user", "content": "Tell me a joke"}]
        async for chunk in client.chat_stream(messages=messages):
            print(chunk["chunk"], end="", flush=True)
            if chunk["done"]:
                print(f"\n\nModel: {chunk['model']}, Latency: {chunk['latency_ms']}ms")

asyncio.run(stream_example())
```

**REST API Streaming:**

```python
import requests

# Streaming generate endpoint
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={"prompt": "Write a story", "model": "qwen3:14b-q4_K_M", "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        # Parse Server-Sent Events format
        if line.startswith(b'data: '):
            import json
            chunk = json.loads(line[6:])  # Remove "data: " prefix
            print(chunk['chunk'], end='', flush=True)
            if chunk['done']:
                print(f"\n\nLatency: {chunk['latency_ms']}ms")
```

**TypeScript/JavaScript Streaming:**

```typescript
const response = await fetch("http://localhost:8000/api/v1/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Write a story",
    model: "qwen3:14b-q4_K_M",
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const chunk = JSON.parse(line.slice(6));
      process.stdout.write(chunk.chunk);
      if (chunk.done) {
        console.log(`\n\nLatency: ${chunk.latency_ms}ms`);
      }
    }
  }
}
```

**Streaming Format:**

- Uses Server-Sent Events (SSE) format: `data: {json}\n\n`
- Each chunk contains: `chunk` (text), `done` (boolean), `model`, `request_id`
- Final chunk (when `done=True`) includes metrics: `latency_ms`, `model_load_ms`, `model_warm_start`, token counts

### Request Queue Management

The REST API includes intelligent request queuing to handle traffic spikes gracefully. Instead of immediately rejecting requests with 503 errors when capacity is reached, the queue system allows requests to wait for available slots.

**Configuration:**

- **Max Concurrent**: 3 requests processed simultaneously
- **Max Queue Size**: 50 requests can wait in queue
- **Default Timeout**: 60 seconds wait time before timeout

**How it Works:**

1. When all 3 processing slots are busy, new requests enter the queue
2. Requests wait for an available slot (up to 60 seconds by default)
3. Once a slot opens, the next queued request begins processing
4. If the queue fills (50+ waiting), new requests are rejected with clear error messages

**Queue Statistics Endpoint:**

Monitor queue health in real-time:

```bash
curl http://localhost:8000/api/v1/queue/stats
```

Response:

```json
{
  "queued": 2,              // Currently waiting
  "in_progress": 3,         // Currently processing
  "completed": 145,         // Total completed
  "failed": 2,              // Total failed
  "rejected": 0,            // Rejected (queue full)
  "timeout": 0,             // Timed out in queue
  "total_wait_time_ms": 15420.5,  // Total wait time
  "max_wait_time_ms": 3250.2,     // Maximum wait observed
  "avg_wait_time_ms": 106.3,      // Average wait time
  "max_concurrent": 3,      // Config: max concurrent
  "max_queue_size": 50,     // Config: max queue size
  "default_timeout": 60.0   // Config: timeout (seconds)
}
```

**Benefits:**

- ✅ **Graceful Degradation**: Requests wait instead of failing immediately
- ✅ **Better UX**: Users see progress instead of errors during traffic spikes
- ✅ **Visibility**: Real-time queue metrics for monitoring
- ✅ **Automatic Cleanup**: Failed requests don't block the queue
- ✅ **Timeout Protection**: Prevents indefinite waiting

**Example Monitoring Script:**

```python
import requests
import time

def monitor_queue():
    while True:
        response = requests.get("http://localhost:8000/api/v1/queue/stats")
        stats = response.json()

        print(f"Queue Status: {stats['queued']} waiting, "
              f"{stats['in_progress']} processing, "
              f"avg wait: {stats['avg_wait_time_ms']:.1f}ms")

        if stats['rejected'] > 0:
            print(f"⚠️  {stats['rejected']} requests rejected (queue full)")

        time.sleep(5)

monitor_queue()
```

### Alternative: Using the Shared Client Library (Python Only)

**Best for**: Python projects that want direct library access

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")

from shared_ollama import SharedOllamaClient
from shared_ollama.core.utils import ensure_service_running

# Automatic service discovery from environment
ensure_service_running()

client = SharedOllamaClient()
response = client.generate("Your prompt here")
```

### Project-Specific Integration

#### Knowledge Machine

Update `Knowledge_Machine/config/main.py`:

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")
from shared_ollama import OllamaConfig, SharedOllamaClient

# Or configure via environment
import os
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
default_model = "qwen3:14b-q4_K_M"  # or "qwen3-vl:8b-instruct-q4_K_M" for VLM tasks

client = SharedOllamaClient(OllamaConfig(
    base_url=ollama_base_url,
    default_model=default_model
))
```

#### Course Intelligence Compiler

Update `Course_Intelligence_Compiler/config/rag_config.yaml`:

```yaml
generation:
  ollama:
    base_url: "http://localhost:11434"  # Or use OLLAMA_BASE_URL env var
    model: "qwen3:14b-q4_K_M"  # or "qwen3-vl:8b-instruct-q4_K_M" for VLM tasks
```

#### Story Machine

Update `Story_Machine/src/story_machine/core/config.py`:

```python
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")
from shared_ollama import SharedOllamaClient
from shared_ollama.core.utils import get_ollama_base_url

# Auto-discover from environment
base_url = get_ollama_base_url()
model = "qwen3:14b-q4_K_M"  # or "qwen3-vl:8b-instruct-q4_K_M" for VLM tasks

client = SharedOllamaClient()
```

See [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) for detailed project-specific examples and migration instructions.

## Configuration

### Environment Variables

**For Client Projects**:

```bash
# Recommended: Use REST API (port 8000)
export API_BASE_URL="http://localhost:8000"

# Alternative: Direct Ollama access (port 11434)
# Only if you need direct Ollama API access (bypasses REST API features)
export OLLAMA_BASE_URL="http://localhost:11434"
```

**For Service Configuration** (auto-detected from `config/model_profiles.yaml`):

**Automatic Configuration Generation** (Recommended):

```bash
# Inspect detected hardware profile and recommended settings
./scripts/generate_optimal_config.sh

# This script:
# - Detects CPU, GPU, RAM, chip type
# - Calculates optimal OLLAMA_MAX_RAM based on model requirements
# - Reserves memory for RAG systems (default: 8GB)
# - Configures GPU acceleration (Metal for Apple Silicon)
# - Sets optimal parallel model count
# - Configures network access (default: localhost for security)
# - Prints the selected model profile (no .env file required)
```

**Override Configuration** (optional):

```bash
# Network Configuration (default: localhost for security)
export OLLAMA_HOST=localhost        # localhost = local only, 0.0.0.0 = network accessible
export API_HOST=0.0.0.0            # REST API host (can be network accessible)

# Memory Configuration (auto-calculated by generate_optimal_config.sh)
export OLLAMA_MAX_RAM=44GB         # Based on model requirements + RAG reserves
export OLLAMA_NUM_PARALLEL=3       # Number of parallel models (max 3)

# GPU Configuration (auto-detected for Apple Silicon)
export OLLAMA_METAL=1              # Metal acceleration (Apple Silicon)
export OLLAMA_NUM_GPU=-1           # Use all GPU cores
export OLLAMA_NUM_THREAD=14        # CPU threads

# Model Keep-Alive
export OLLAMA_KEEP_ALIVE=30m       # How long models stay loaded after last use
export OLLAMA_DEBUG=false
```

**Network Access:**

- **Default**: `OLLAMA_HOST=localhost` (local access only, more secure)
- **Network Access**: Set `OLLAMA_HOST=0.0.0.0` to allow connections from other machines on the network
- **REST API**: `API_HOST=0.0.0.0` allows network access to the REST API (port 8000) while keeping Ollama local-only

The client automatically discovers the service URL from these environment variables, so projects don't need hardcoded URLs.

### Keep-Alive Configuration

The `OLLAMA_KEEP_ALIVE` setting determines how long models remain in memory after the last request before being automatically unloaded.

**Trade-offs:**

| Keep-Alive Time | Pros | Cons | Best For |
|----------------|------|------|----------|
| **1-2 minutes** | Frees memory quickly, allows faster model switching | More frequent reloads, slower responses after idle periods | Memory-constrained environments, infrequent use |
| **5 minutes** (current) | Balanced - reasonable response time without excessive memory use | Model unloaded after 5 min idle | **General use, development environments** |
| **15-30 minutes** | Fewer reloads, faster responses for active sessions | Models stay in memory longer, blocks other models | Active development, frequent model switching |
| **Infinite** (`-1` or `0`) | Never unloads, fastest response time | Always uses memory, blocks other models | Production with dedicated models, high-traffic |

**Recommendations:**

- **Development/Testing**: `5m` (current) - Good balance for intermittent use
- **Active Development**: `15m-30m` - Reduces reloads during active coding sessions
- **Memory-Constrained**: `2m-3m` - Faster memory release
- **Production (Single Model)**: `15m-30m` or `0` - Keep model hot for consistent performance
- **Production (Multiple Models)**: `5m-10m` - Balance between performance and memory efficiency

**To change keep-alive:**

```bash
export OLLAMA_KEEP_ALIVE=15m
./scripts/shutdown.sh && ./scripts/start.sh
```

### Port Configuration

- **REST API Port**: 8000 (FastAPI service)
- **Ollama Port**: 11434 (managed internally by REST API)
- **REST API Endpoint**: `http://localhost:8000/api/v1/generate`
- **REST API Docs**: `http://localhost:8000/api/docs`

## Health Checks

### Automated Health Check

```bash
# Comprehensive setup verification (recommended)
./scripts/verify_setup.sh

# This checks:
# - Detects hardware profile and loads recommended settings
# - Ollama installation
# - Service status
# - Model availability
# - Downloads missing models
# - Health checks (tests all models)

# Quick health check
./scripts/health_check.sh

# This checks:
# - Ollama service is running
# - All models are available
# - API is responding

### Manual Testing

```bash
# Health check (REST API)
curl http://localhost:8000/api/v1/health

# List models (REST API)
curl http://localhost:8000/api/v1/models

# Test generation (REST API)
curl http://localhost:8000/api/v1/generate -d '{
  "model": "qwen3:14b-q4_K_M",
  "prompt": "Why is the sky blue?"
}'
```

## Model Management

### List Loaded Models

```bash
ollama list
```

### Remove a Model

```bash
ollama rm model_name
```

### Update Models

```bash
# Pull latest version
ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama pull qwen3:14b-q4_K_M
```

## Service Management

**Start the Service:**

```bash
# Start REST API (automatically manages Ollama internally)
./scripts/start.sh

# The REST API will:
# - Auto-detect system hardware + model profile
# - Apply optimal configuration (no .env required)
# - Start Ollama with optimizations
# - Provide REST API on port 8000
```

**Stop the Service:**

```bash
# Stop REST API and Ollama
./scripts/shutdown.sh
```

**Check Status:**

```bash
# Quick status check
./scripts/status.sh

# Health check (REST API)
curl http://localhost:8000/api/v1/health

# View logs
tail -f logs/api.log
```

## Performance Optimizations

### Current Optimizations

✅ **Fully Optimized for Apple Silicon MPS (Metal Performance Shaders):**

- ✅ **MPS/Metal GPU**: Explicitly enabled (`OLLAMA_METAL=1`)
- ✅ **GPU Cores**: All Metal GPU cores utilized (`OLLAMA_NUM_GPU=-1`)
- ✅ **CPU**: All 10 cores automatically detected and utilized
- ✅ **Threading**: Auto-detected based on CPU cores
- ✅ **Memory**: Efficient allocation for both models simultaneously
- ✅ **Performance**: Maximum GPU acceleration via Metal/MPS

### Hardware Detection

The service auto-detects hardware:

- **Apple Silicon** (M1/M2/M3/M4): Metal/MPS GPU acceleration (explicitly enabled)
- **CPU**: All cores automatically detected and utilized (10 cores)
- **Memory**: Efficient allocation based on model requirements
- **GPU**: All Metal GPU cores available for inference

### Apple Silicon MPS Optimization

**Explicit MPS Configuration:**

- ✅ `OLLAMA_METAL=1`: Explicitly enables Metal/MPS acceleration
- ✅ `OLLAMA_NUM_GPU=-1`: Uses all available Metal GPU cores
- ✅ `OLLAMA_NUM_THREAD`: Auto-detected (matches 10 CPU cores)

**Performance Benefits:**

- Maximum GPU utilization via Metal Performance Shaders
- Faster inference times with full GPU acceleration
- Efficient memory management for concurrent model execution
- Optimal resource allocation for Apple Silicon architecture

**To verify MPS is active:**

```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep -i metal

# Monitor GPU usage (in Activity Monitor or with system_profiler)
```

## Troubleshooting

### Service Won't Start

```bash
# Check if Ollama is installed
which ollama

# Start REST API
./scripts/start.sh

# Check if REST API is running
curl http://localhost:8000/api/v1/health

# View logs
tail -f logs/api.log
```

### Models Not Found

```bash
# Pull models
ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama pull qwen3:14b-q4_K_M

# Verify
ollama list
```

### Connection Refused

```bash
# Check if REST API is running
curl http://localhost:8000/api/v1/health

# If not running, start it
./scripts/start.sh

# If already running but not responding, restart
./scripts/shutdown.sh && ./scripts/start.sh
```

## Security

### Network Isolation

The Ollama service is **not exposed to the internet** by default. Only localhost connections are allowed.

**Default Configuration:**

- `OLLAMA_HOST=localhost` - Ollama service accessible only from localhost (secure default)
- `API_HOST=0.0.0.0` - REST API can be network-accessible (port 8000) if needed

**To Enable Network Access:**

- Export `OLLAMA_HOST=0.0.0.0` before launching the service to allow connections from other machines on the same network
- Run `./scripts/generate_optimal_config.sh` to confirm the active configuration
- **Security Note**: Only enable network access on trusted networks. Consider firewall rules for untrusted networks.

### Model Access Control

Models are stored locally: `~/.ollama/models`

## Monitoring

### Quick Status Check

```bash
# Fast status overview (recommended)
./scripts/status.sh
```

Shows:

- Service health
- Available models
- Process information
- Memory usage
- Quick health test

### Logs

```bash
# View REST API logs
tail -f logs/api.log

# View error logs
tail -f logs/api.error.log

# View structured request log (JSON lines)
tail -f logs/requests.jsonl

# View performance logs
tail -f logs/performance.jsonl
```

### Metrics & Performance

**Request Metrics** (via monitoring):

- Overall latency (p50, p95, p99)
- Success/failure rates
- Usage by model and operation

**REST API Logs**:

- API request logs: `logs/api.log`
- Error logs: `logs/api.error.log`
- Performance logs: `logs/performance.jsonl` (detailed performance metrics)
- Structured request events (model timings, load durations): `logs/requests.jsonl`

**Performance Analysis**:

```bash
# View performance report
python scripts/performance_report.py

# Filter by model
python scripts/performance_report.py --model qwen3:14b-q4_K_M

# Last hour
python scripts/performance_report.py --window 60
```

**Quick Monitoring**:

- **Quick status**: `./scripts/status.sh` (fast overview)
- **Health checks**: `./scripts/health_check.sh` (comprehensive)
- **Model status**: `curl http://localhost:8000/api/v1/models`
- **Resource usage**: `top -pid $(pgrep ollama)` or Activity Monitor

**Note**: See `PERFORMANCE_MONITORING.md` for detailed performance tracking capabilities.

### API Monitoring Endpoints

The service provides comprehensive monitoring endpoints for metrics, performance, and analytics:

#### Service Metrics (`GET /api/v1/metrics`)

Get comprehensive service metrics including request counts, latency statistics, and error breakdowns:

```bash
# Get all metrics
curl http://localhost:8000/api/v1/metrics

# Get metrics from last hour
curl "http://localhost:8000/api/v1/metrics?window_minutes=60"
```

**Response includes**:

- Total requests, successful/failed counts
- Latency percentiles (P50, P95, P99)
- Requests by model and operation
- Error breakdowns by type
- First/last request timestamps

#### Performance Statistics (`GET /api/v1/performance/stats`)

Get detailed performance metrics including token generation rates and timing breakdowns:

```bash
curl http://localhost:8000/api/v1/performance/stats
```

**Response includes**:

- Average tokens per second (generation throughput)
- Average model load time
- Average generation time
- Per-model breakdowns with request counts

#### Analytics (`GET /api/v1/analytics`)

Get comprehensive analytics with project-level tracking and time-series data:

```bash
# Get all analytics
curl http://localhost:8000/api/v1/analytics

# Get analytics for specific project
curl "http://localhost:8000/api/v1/analytics?project=Docling_Machine"

# Get analytics from last hour
curl "http://localhost:8000/api/v1/analytics?window_minutes=60"

# Combined filters
curl "http://localhost:8000/api/v1/analytics?window_minutes=60&project=Docling_Machine"
```

**Response includes**:

- Total requests, success rates
- Latency percentiles (P50, P95, P99)
- Requests by model, operation, and project
- Project-level metrics (detailed breakdowns per project)
- Hourly time-series metrics
- Time range (start_time, end_time)

**Project Tracking**: Projects are identified via the `X-Project-Name` header in requests. Analytics automatically tracks usage by project.

#### Queue Statistics (`GET /api/v1/queue/stats`)

Get real-time queue performance metrics:

```bash
curl http://localhost:8000/api/v1/queue/stats
```

**Response includes**:

- Current queue state (queued, in_progress)
- Historical counts (completed, failed, rejected, timeout)
- Wait time statistics (total, max, average)
- Configuration (max_concurrent, max_queue_size, timeout)

### Performance Data Collection

The service collects comprehensive performance data:

**Structured Logs** (`logs/requests.jsonl`):

- Request-level metrics (latency, model, operation, status)
- Model load times (`model_load_ms`)
- Warm start indicators (`model_warm_start`)
- Project names (from `X-Project-Name` header)
- Client IP addresses
- Error information

**Performance Logs** (`logs/performance.jsonl`):

- Detailed timing breakdowns (load, prompt eval, generation)
- Token-level metrics (tokens/second, prompt tokens/second)
- Per-model performance statistics
- Comprehensive performance data for analysis

**In-Memory Metrics**:

- Real-time aggregations (via `/api/v1/metrics`)
- Project-based analytics (via `/api/v1/analytics`)
- Performance statistics (via `/api/v1/performance/stats`)

**See**: `docs/PERFORMANCE_DATA_COLLECTED.md` for complete details on all collected metrics.

## Cost and Resource Management

### Memory Usage

**Memory Usage:**

- `qwen3-vl:8b-instruct-q4_K_M`: ~6 GB RAM when loaded (laptop profile)
- `qwen3:14b-q4_K_M`: ~8 GB RAM when loaded (laptop profile)
- `qwen3-vl:32b`: ~21 GB RAM when loaded (workstation profile)
- `qwen3:30b`: ~19 GB RAM when loaded (workstation profile)
- **Laptop total**: ~14 GB when both models loaded
- **Workstation total**: ~40 GB when both models loaded
- **Up to 3 models can run simultaneously** on machines that meet the RAM requirement (configurable via `OLLAMA_NUM_PARALLEL`)

**Behavior**: Models are automatically loaded when requested and unloaded after 5 minutes of inactivity. Both models can be active at the same time if needed, reducing switching delays.

**Memory Configuration:**

The service uses **model-based memory calculation** (not percentage-based) to ensure adequate memory for RAG systems and other services running on the same machine:

- **Calculation**: Based on actual model requirements (largest model × parallel count + inference buffer + overhead)
- **Reserves**: Automatically reserves memory for:
  - System overhead: 8GB
  - RAG systems: 8GB (configurable via `RAG_RESERVE_GB` environment variable)
  - Safety buffer: 4GB
- **Automatic Configuration**: Run `./scripts/generate_optimal_config.sh` to auto-detect hardware and print the optimal settings (applied dynamically, no `.env` file)
- **Customization**: Adjust RAG reserve if needed:

  ```bash
  export RAG_RESERVE_GB=12  # For larger RAG systems
  ./scripts/calculate_memory_limit.sh
  ```

See `scripts/calculate_memory_limit.sh` for detailed memory calculation logic.

### Performance

**Without Warm-up:**

- **First request**: ~2-3 seconds (model loading into memory)
- **Subsequent requests**: ~100-500ms (depends on prompt length)

**With Warm-up (Models Pre-loaded):**

- **First request**: ~100-500ms (model already in memory)
- **Subsequent requests**: ~100-500ms (consistently fast)

See [Warm-up & Pre-loading](#warm-up--pre-loading) section below for setup.

## Warm-up & Pre-loading

### Pre-download Models

Ensure all models are downloaded locally before first use:

```bash
# Pre-download all required models
./scripts/preload_models.sh
```

This verifies models are available locally, eliminating download delays during inference.

### Warm-up Models (Pre-load into Memory)

Pre-load models into memory to eliminate first-request latency:

```bash
# Warm up models (loads them into memory)
./scripts/warmup_models.sh

# Or warm up with custom keep-alive duration
KEEP_ALIVE=60m ./scripts/warmup_models.sh
```

**What this does:**

- Sends minimal requests to each model
- Loads models into GPU/CPU memory
- Sets keep-alive to keep models loaded (default: 30 minutes)
- Reduces first-request latency from ~2-3s to ~100-500ms

### Automated Warm-up on Startup

#### Option 1: Warm-up after service start

```bash
# Start REST API
./scripts/start.sh

# In another terminal, warm up models (optional)
./scripts/warmup_models.sh
```

#### Option 2: Warm-up via REST API

```bash
# Warm up a specific model via REST API
curl http://localhost:8000/api/v1/generate -d '{
  "model": "qwen3:14b-q4_K_M",
  "prompt": "Hi",
  "options": {"num_predict": 1},
  "keep_alive": "30m"
}'
```

### Keep-Alive Recommendations for Production

**High-Traffic Production:**

```bash
# Keep models loaded indefinitely
KEEP_ALIVE=-1 ./scripts/warmup_models.sh
```

**Development/Testing:**

```bash
# Keep models loaded for active session
KEEP_ALIVE=30m ./scripts/warmup_models.sh
```

**Memory-Constrained:**

```bash
# Shorter keep-alive, models reload as needed
KEEP_ALIVE=10m ./scripts/warmup_models.sh
```

### Performance Comparison

| Setup | First Request | Subsequent Requests | Best For |
|-------|--------------|---------------------|----------|
| **No warm-up** | 2-3 seconds | 100-500ms | Development, occasional use |
| **Warm-up (30m keep-alive)** | 100-500ms | 100-500ms | Active development |
| **Warm-up (infinite keep-alive)** | 100-500ms | 100-500ms | Production, high-traffic |

## New Features & Enhancements

### Async/Await Support

Modern async/await client for asynchronous Python applications:

```python
import asyncio
from shared_ollama import AsyncSharedOllamaClient

async def main():
    async with AsyncSharedOllamaClient() as client:
        response = await client.generate("Hello!")
        print(response.text)

asyncio.run(main())
```

**Installation**: `pip install -e ".[async]"`

### Monitoring & Metrics

**✅ FULLY ENABLED** - Comprehensive monitoring and analytics:

**Via API** (Recommended):

```bash
# Service metrics
curl http://localhost:8000/api/v1/metrics

# Performance statistics
curl http://localhost:8000/api/v1/performance/stats

# Analytics (with project tracking)
curl http://localhost:8000/api/v1/analytics
curl "http://localhost:8000/api/v1/analytics?project=Docling_Machine&window_minutes=60"

# Queue statistics
curl http://localhost:8000/api/v1/queue/stats
```

**Via Python**:

```python
from shared_ollama import MetricsCollector, track_request

# Track a request
with track_request("qwen3-vl:8b-instruct-q4_K_M", "generate"):
    response = client.generate("Hello!")

# Get metrics
metrics = MetricsCollector.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
print(f"P95 latency: {metrics.p95_latency_ms:.2f}ms")
```

**See**: [API Monitoring Endpoints](#api-monitoring-endpoints) section for complete details.

### Enhanced Resilience

Automatic retry with exponential backoff and circuit breaker:

```python
from resilience import ResilientOllamaClient

client = ResilientOllamaClient()
response = client.generate("Hello!")  # Automatic retry & circuit breaker
```

### Testing

Comprehensive test suite for reliability:

```bash
# Run tests
pytest

# With coverage
pytest --cov
```

See `IMPLEMENTED_ENHANCEMENTS.md` for full details.

### Enhanced Analytics

**✅ NOW FULLY ENABLED** - Project-based analytics with time-series analysis:

**Via API** (Recommended):

```bash
# Get all analytics
curl http://localhost:8000/api/v1/analytics

# Get analytics for specific project
curl "http://localhost:8000/api/v1/analytics?project=Docling_Machine"

# Get analytics from last hour
curl "http://localhost:8000/api/v1/analytics?window_minutes=60"

# Combined filters
curl "http://localhost:8000/api/v1/analytics?window_minutes=60&project=Docling_Machine"
```

**Via Python**:

```python
from shared_ollama import AnalyticsCollector, get_analytics_json

# Get analytics report
analytics = get_analytics_json(window_minutes=60, project="Docling_Machine")
print(f"Total requests: {analytics['total_requests']}")
print(f"Success rate: {analytics['success_rate']:.2%}")
print(f"Average latency: {analytics['average_latency_ms']:.2f}ms")
print(f"P95 latency: {analytics['p95_latency_ms']:.2f}ms")

# Export analytics
AnalyticsCollector.export_json("analytics.json")
AnalyticsCollector.export_csv("analytics.csv")
```

**Project Tracking**: Automatically enabled! Include `X-Project-Name` header in requests:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    headers={"X-Project-Name": "Docling_Machine"},
    json={"model": "qwen3:14b-q4_K_M", "messages": [...]}
)
```

**Features**:

- ✅ Project-level usage tracking (automatic with `X-Project-Name` header)
- ✅ Hourly time-series metrics
- ✅ Latency percentiles (P50, P95, P99)
- ✅ Success rates and error breakdowns
- ✅ JSON/CSV export capabilities
- ✅ Time-window filtering
- ✅ Per-project detailed metrics

**See**: [API Monitoring Endpoints](#api-monitoring-endpoints) section for complete API documentation.

### API Documentation

Complete API reference and OpenAPI specification:

- **API Reference**: See `docs/API_REFERENCE.md`
- **OpenAPI Spec**: `docs/openapi.yaml` (OpenAPI 3.1.0)
- **View Interactive Docs**: Use Swagger UI or online editor

### Type Stubs

Full type stubs for better IDE support:

```bash
# Type stubs are automatically included
# Your IDE should detect them automatically
```

### CI/CD

GitHub Actions workflows for automated testing and releases:

- **CI Workflow**: `.github/workflows/ci.yml` - Tests, linting, type checking
- **Release Workflow**: `.github/workflows/release.yml` - Automated releases

## Code Quality & Modernization

### Python 3.13+ Modernization

This codebase has been fully modernized to leverage Python 3.13+ native features:

**Native Features:**
§s

- ✅ `datetime.now(UTC)` for timezone-aware timestamps
- ✅ `time.perf_counter()` for precise performance measurements
- ✅ Modern type hints with `|` union syntax
- ✅ `collections.abc.Generator` for context managers

**Code Quality:**

- ✅ 90%+ type hint coverage across all modules
- ✅ Comprehensive error handling with specific exception types
- ✅ JSON validation and response structure checking
- ✅ File I/O error handling with detailed logging
- ✅ DRY principles with extracted helper functions

**Testing:**

- ✅ **33+ comprehensive tests** (all passing) for API endpoints
- ✅ **Reusable test infrastructure** with helper utilities and fixtures
- ✅ **Behavioral testing** focused on real-world scenarios
- ✅ **Edge case coverage** including validation, errors, and timeouts
- ✅ **Dependency injection testing** with FastAPI TestClient
- ✅ **Full test suite** covering client, async client, resilience, telemetry, and utilities

See [docs/TESTING_PLAN.md](docs/TESTING_PLAN.md) for comprehensive testing strategy and [docs/TESTING_IMPLEMENTATION_SUMMARY.md](docs/TESTING_IMPLEMENTATION_SUMMARY.md) for implementation details.

See recent commits for detailed modernization improvements.

## Architecture

This project follows **Clean Architecture** principles with strict dependency rules:

### Layer Structure

1. **Domain Layer** (`domain/`): Pure business logic with no external dependencies
   - Entities: `Model`, `ModelInfo`, `GenerationRequest`, `ChatCompletionRequest`
   - Value Objects: `ModelName`, `Prompt`, `SystemMessage`
   - Domain Exceptions: `InvalidModelError`, `InvalidPromptError`, `InvalidRequestError`

2. **Application Layer** (`application/`): Orchestrates domain logic
   - Use Cases: `GenerateUseCase`, `ChatUseCase`, `ListModelsUseCase`
   - Interfaces: `OllamaClientInterface`, `RequestLoggerInterface`, `MetricsCollectorInterface`

3. **Infrastructure Layer** (`infrastructure/`): External service implementations
   - Adapters: `AsyncOllamaClientAdapter`, `RequestLoggerAdapter`, `MetricsCollectorAdapter`

4. **Interface Adapters** (`api/`): HTTP/FastAPI layer
   - Controllers: FastAPI endpoints (thin, no business logic)
   - Mappers: Convert between API models and domain entities
   - Dependencies: Dependency injection for use cases

### Key Benefits

- ✅ **Testability**: All dependencies injected, easy to mock
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Flexibility**: Swap implementations without changing business logic
- ✅ **Type Safety**: Protocol-based interfaces with full type hints

For detailed architecture documentation, see:

- [Clean Architecture Refactoring](docs/CLEAN_ARCHITECTURE_REFACTORING.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

## Development

This project uses modern Python 3.13+ tooling for development:

### Development Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Code Quality Tools

**Ruff** - Fast linter and formatter (replaces Black, isort, flake8, etc.)

- **Format code**: `ruff format .`
- **Lint code**: `ruff check .`
- **Auto-fix**: `ruff check --fix .`

**Pyright** - Static type checker (Microsoft's type checker)

- **Type check**: `pyright src/shared_ollama`

**Testing** - pytest with comprehensive test suite

- **Run all tests**: `pytest`
- **Run API tests**: `pytest tests/test_api_server.py -v`
- **With coverage**: `pytest --cov`
- **Test utilities**: See `tests/helpers.py` for reusable test components

**Test Infrastructure:**

- ✅ 33+ passing tests covering all endpoints and scenarios
- ✅ Reusable fixtures and helper functions
- ✅ FastAPI TestClient for reliable dependency injection testing
- ✅ Comprehensive error path and edge case coverage

See [docs/TESTING_PLAN.md](docs/TESTING_PLAN.md) for testing strategy and [docs/TESTING_IMPLEMENTATION_SUMMARY.md](docs/TESTING_IMPLEMENTATION_SUMMARY.md) for implementation details.

### Makefile Commands

```bash
make lint        # Run Ruff linter
make format      # Format code with Ruff
make type-check  # Run Pyright type checker
make test        # Run tests with pytest
make check       # Run all checks (lint, format, type-check)
make fix         # Auto-fix linting issues
make all         # Clean, install, format, fix, type-check, test
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run Ruff and Pyright before each commit.

### Configuration Files

- **`pyproject.toml`** - Project configuration, dependencies, Ruff, and Pyright settings
- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration
- **`Makefile`** - Convenient development commands

### Python Version

This project requires **Python 3.13+** and uses modern Python features:

- Native type annotations (`list` instead of `List`, `dict` instead of `Dict`)
- Union types (`X | None` instead of `Optional[X]`)
- Modern enum patterns
- Latest typing features

## Contributing

When adding new models or modifying the service:

1. Update installation scripts if needed
2. Update preload/warmup scripts with new models
3. Update this README
4. Test with all projects
5. Document model size and use cases

## License

MIT

## Support

For issues or questions:

- Check logs: `tail -f logs/api.log`
- Run health check: `./scripts/health_check.sh`
- Verify service: `curl http://localhost:8000/api/v1/health`
- Open issue in project repository
