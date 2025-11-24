# POML Examples for Shared Ollama Service

This directory contains working examples of using POML (Prompt Orchestration Markup Language) with the Shared Ollama Service.

## Files

### POML Templates

- **`chat_with_tools.poml`** - Tool/function calling template with multiple tools
- **`extract_event.poml`** - JSON schema output validation example
- **`analyze_image.poml`** - VLM (Vision Language Model) with structured output

### Python Examples

- **`weather_assistant.py`** - Complete working example with tool calling

## Prerequisites

```bash
# Install POML
pip install poml

# Install requests for HTTP calls
pip install requests

# Ensure Shared Ollama Service is running
./scripts/core/start.sh
```

## Running the Examples

### 1. Weather Assistant (Tool Calling)

```bash
cd examples/poml
python weather_assistant.py
```

This example demonstrates:
- Loading POML templates
- Calling tools/functions
- Handling tool responses
- Conversation flow with tool results

### 2. Event Extraction (JSON Schema)

```python
import poml
import requests
import json

context = {
    "text": "Alice and Bob are meeting at the Science Fair on Friday in Building A."
}
params = poml.poml("extract_event.poml", context=context, format="openai_chat")

response = requests.post(
    "http://0.0.0.0:8000/api/v1/chat",
    json=params
).json()

event = json.loads(response["message"]["content"])
print(event)
```

### 3. Image Analysis (VLM + JSON Schema)

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

# Add images for VLM endpoint
request_data = {
    **params,
    "images": [f"data:image/jpeg;base64,{img_data}"]
}

response = requests.post(
    "http://0.0.0.0:8000/api/v1/vlm",
    json=request_data
).json()

analysis = json.loads(response["message"]["content"])
print(analysis)
```

## POML Features Demonstrated

### 1. Tool Definitions

```xml
<tool-definition name="get_weather" description="Get weather">
{
  "type": "object",
  "properties": {
    "location": {"type": "string"}
  }
}
</tool-definition>
```

### 2. Tool Requests/Responses

```xml
<tool-request if="tool_request" id="{{ tool_request.id }}"
              name="{{ tool_request.name }}"
              parameters="{{ tool_request.parameters }}" />

<tool-response if="tool_response" id="{{ tool_response.id }}"
               name="{{ tool_response.name }}">
  <object data="{{ tool_response.result }}"/>
</tool-response>
```

### 3. JSON Schema Output

```xml
<output-schema>
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "date": {"type": "string"}
  },
  "required": ["name", "date"]
}
</output-schema>
```

### 4. Runtime Parameters

```xml
<runtime model="qwen3:14b-q4_K_M" temperature="0.7" max-tokens="500"/>
```

### 5. Template Variables

```xml
<human-msg>{{ user_question }}</human-msg>
<task>{{ task_description }}</task>
```

## Learn More

- **POML Documentation**: https://github.com/microsoft/poml
- **Shared Ollama Service Docs**: See main README.md
- **API Reference**: http://0.0.0.0:8000/api/docs (when service is running)

## Troubleshooting

### "Connection refused"
Make sure the Shared Ollama Service is running:
```bash
./scripts/core/start.sh
./scripts/core/status.sh
```

### "Tool not found" errors
The `weather_assistant.py` example uses simulated functions. In production:
1. Replace `get_weather()` with a real weather API
2. Replace `calculate()` with a proper math parser (e.g., `sympy`)

### POML import errors
Install POML:
```bash
pip install poml
```

## Next Steps

1. Modify the POML templates to add your own tools
2. Create custom JSON schemas for your use case
3. Combine tools with VLM for vision-powered assistants
4. Use `<include>` tags to reuse common POML components
