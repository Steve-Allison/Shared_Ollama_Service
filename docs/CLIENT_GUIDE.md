# API Client Quickstart Guide

This guide provides examples for quickly interacting with the Shared Ollama Service API using `curl`, Python, TypeScript, and Go.

Before you begin, ensure the API service is running:

```bash
./scripts/core/start.sh
```

## Text-Only Chat

### Native Ollama Format (`/api/v1/chat`)

Send a text-only message to a language model using native Ollama format.

**Request:**

```bash
curl -X POST http://0.0.0.0:8000/api/v1/chat \
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

### OpenAI-Compatible Format (`/api/v1/chat/completions`)

Send a text-only message using OpenAI-compatible format. Perfect for OpenAI clients and libraries.

**Request:**

```bash
curl -X POST http://0.0.0.0:8000/api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen3:14b-q4_K_M",
           "messages": [
             {"role": "user", "content": "Tell me a short story about a brave knight."}
           ]
         }'
```

**Example Response (OpenAI-Compatible JSON):**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": ...,
  "model": "qwen3:14b-q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Sir Reginald, a knight known for his polka-dotted shield..."
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

## VLM with Images

### Native Ollama Format (`/api/v1/vlm`)

Send a multimodal request with text and an image using Ollama's native format.

**Request:**

```bash
# First, convert your image to a base64 data URL
# Example using Python:
# python -c "import base64; print('data:image/jpeg;base64,' + base64.b64encode(open('photo.jpg', 'rb').read()).decode('utf-8'))"
IMAGE_DATA_URL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAA..." # Replace with your actual base64 image data

curl -X POST http://0.0.0.0:8000/api/v1/vlm \
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

### OpenAI-Compatible Format (`/api/v1/vlm/openai`)

Send a multimodal request with text and an image using an OpenAI-compatible message format.

**Request:**

```bash
# First, convert your image to a base64 data URL
# Example using Python:
# python -c "import base64; print('data:image/jpeg;base64,' + base64.b64encode(open('photo.jpg', 'rb').read()).decode('utf-8'))"
IMAGE_DATA_URL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAA..." # Replace with your actual base64 image data

curl -X POST http://0.0.0.0:8000/api/v1/vlm/openai \
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

## Python Examples

### Basic Text Chat

```python
import requests

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    json={
        "model": "qwen3:14b-q4_K_M",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(response.json()["message"]["content"])
```

### VLM with Images

```python
import requests
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Native Ollama format
response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "messages": [
            {"role": "user", "content": "What's in this image?"}
        ],
        "images": [f"data:image/jpeg;base64,{img_data}"]
    }
)
print(response.json()["message"]["content"])
```

### Streaming Responses

```python
import requests
import json

response = requests.post(
    "http://0.0.0.0:8000/api/v1/generate",
    json={"prompt": "Write a story", "model": "qwen3:14b-q4_K_M", "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        chunk = json.loads(line[6:])
        print(chunk['chunk'], end='', flush=True)
        if chunk['done']:
            print(f"\n\nLatency: {chunk['latency_ms']}ms")
```

## TypeScript/JavaScript Examples

### Basic Text Chat

```typescript
const response = await fetch("http://0.0.0.0:8000/api/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "qwen3:14b-q4_K_M",
    messages: [
      { role: "user", content: "Hello!" }
    ]
  })
});

const data = await response.json();
console.log(data.message.content);
```

### VLM with Images

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

### Streaming Responses

```typescript
const response = await fetch("http://0.0.0.0:8000/api/v1/generate", {
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

## Go Examples

### Basic Text Chat

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    requestBody := map[string]interface{}{
        "model": "qwen3:14b-q4_K_M",
        "messages": []map[string]string{
            {"role": "user", "content": "Hello!"},
        },
    }

    jsonData, _ := json.Marshal(requestBody)
    resp, err := http.Post(
        "http://0.0.0.0:8000/api/v1/chat",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    fmt.Println(result)
}
```

## Connecting from Remote Clients

The examples above show `0.0.0.0:8000` which works for local connections. To connect from **another machine** on your network:

### 1. Find the Server's IP Address

```bash
# On macOS/Linux
ifconfig | grep "inet "
# Look for your local network IP (e.g., 192.168.1.100)

# Or use hostname
hostname -I  # Linux
ipconfig getifaddr en0  # macOS WiFi
```

### 2. Use the Server's IP in Client Code

**Python:**

```python
import requests

API_BASE_URL = "http://192.168.1.100:8000"

response = requests.post(
    f"{API_BASE_URL}/api/v1/chat",
    json={
        "model": "qwen3:14b-q4_K_M",
        "messages": [{"role": "user", "content": "Hello from remote client!"}]
    }
)
print(response.json()["message"]["content"])
```

**TypeScript/JavaScript:**

```typescript
const API_BASE_URL = process.env.API_BASE_URL || "http://192.168.1.100:8000";

const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "qwen3:14b-q4_K_M",
    messages: [{ role: "user", content: "Hello from remote client!" }]
  })
});
```

### 3. Network Configuration

The REST API binds to `0.0.0.0:8000` by default, making it accessible from the network. The internal Ollama service (`localhost:11434`) remains local-only for security.

If you encounter connection issues:

- **Firewall**: Ensure port 8000 is allowed through your firewall
- **Network**: Verify both machines are on the same network
- **Test Connectivity**: Use `curl http://<server-ip>:8000/api/v1/health` from the client machine

## CORS and Browser Support

The REST API has **CORS (Cross-Origin Resource Sharing) enabled** by default, allowing web browsers and client-side JavaScript to make requests to the service.

**Browser JavaScript Example:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Ollama Chat Client</title>
</head>
<body>
    <h1>Chat with Ollama</h1>
    <textarea id="prompt" placeholder="Enter your message..."></textarea>
    <button onclick="sendMessage()">Send</button>
    <div id="response"></div>

    <script>
        async function sendMessage() {
            const prompt = document.getElementById('prompt').value;
            const responseDiv = document.getElementById('response');

            try {
                const response = await fetch('http://192.168.1.100:8000/api/v1/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Project-Name': 'WebChatClient'
                    },
                    body: JSON.stringify({
                        model: 'qwen3:14b-q4_K_M',
                        messages: [
                            { role: 'user', content: prompt }
                        ]
                    })
                });

                const data = await response.json();
                responseDiv.innerHTML = `<strong>Assistant:</strong> ${data.message.content}`;
            } catch (error) {
                responseDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
            }
        }
    </script>
</body>
</html>
```

## See Also

- [VLM Guide](VLM_GUIDE.md) - Complete vision-language model guide
- [API Reference](API_REFERENCE.md) - Full endpoint documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - Project integration examples

