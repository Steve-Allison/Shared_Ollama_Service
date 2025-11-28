# POML Support (Prompt Orchestration Markup Language)

The Shared Ollama Service **fully supports** [POML](https://github.com/microsoft/poml) for structured prompt engineering and LLM orchestration.

## What is POML?

**POML** (Prompt Orchestration Markup Language) by Microsoft is an XML-based markup language that brings structure and maintainability to LLM prompts. It's like HTML for AI prompts.

## Why Use POML?

- ✅ **Structured Prompts**: Organize complex prompts with clear structure
- ✅ **Tool/Function Calling**: Define tools with `<tool-definition>` tags
- ✅ **Template Variables**: Use `{{ variable }}` for dynamic content
- ✅ **File Inclusion**: Load content with `<document src="file.txt">`
- ✅ **Control Flow**: Use `for` and `if` attributes for logic
- ✅ **JSON Schema Output**: Structured output validation
- ✅ **Runtime Parameters**: Set model params in POML templates

## Supported POML Features

### 1. Tool/Function Calling ⭐ **NEW**

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
    "http://0.0.0.0:8000/api/v1/chat",
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
            "http://0.0.0.0:8000/api/v1/chat",
            json=params
        ).json()

        print(final_response["message"]["content"])
        # "The weather in Paris is currently 22°C and sunny."
```

### 2. JSON Schema Output Validation

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
    "http://0.0.0.0:8000/api/v1/chat",
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

### 3. VLM with POML

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
    "http://0.0.0.0:8000/api/v1/vlm",
    json=request_data
).json()

analysis = json.loads(response["message"]["content"])
print(f"Objects: {analysis['objects']}")
print(f"Scene: {analysis['scene_type']}")
```

### 4. Runtime Parameters

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

## API Endpoints Supporting POML

All generation, chat, and VLM endpoints support POML-generated requests:

| Endpoint | POML Features Supported | Use Case |
|----------|------------------------|----------|
| `/api/v1/generate` | Tools, JSON schema, runtime params | Text generation with function calling |
| `/api/v1/chat` | Tools, JSON schema, runtime params | Text chat with function calling |
| `/api/v1/chat/completions` | Tools, JSON schema, runtime params | OpenAI-compatible text chat with function calling |
| `/api/v1/vlm` | Tools, JSON schema, runtime params, images | Vision + text with tools |
| `/api/v1/vlm/openai` | Tools, JSON schema, runtime params, images | OpenAI-compatible VLM |
| `/api/v1/batch/chat` | All chat features | Batch text processing |
| `/api/v1/batch/chat/completions` | All chat features (OpenAI format) | Batch OpenAI-compatible text processing |
| `/api/v1/batch/vlm` | All VLM features | Batch vision processing |
| `/api/v1/batch/vlm/completions` | All VLM features (OpenAI format) | Batch OpenAI-compatible vision processing |

## Request Format with Tools

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

## Response Format with Tool Calls

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

## Installing POML

```bash
# Install POML Python SDK
pip install poml

# Or with specific integrations
pip install poml[agent]     # AgentOps tracing
pip install poml[mlflow]    # MLflow integration
pip install poml[weave]     # Weights & Biases Weave
```

## POML Resources

- **Official Repository**: <https://github.com/microsoft/poml>
- **Documentation**: <https://github.com/microsoft/poml/tree/main/docs>
- **Examples**: See `examples/poml/` directory in this repository
- **VS Code Extension**: Available in marketplace

## Benefits for Your Workflow

1. **Maintainable Prompts**: Store prompts as `.poml` files, version control them
2. **Reusable Components**: `<include src="common.poml" />` for shared logic
3. **Type-Safe Tools**: JSON schema validation prevents errors
4. **Template Variables**: Dynamic prompts without string concatenation
5. **Documentation**: Self-documenting prompt structure
6. **Testing**: Easier to test prompts with different contexts

## Complete Working Example

See `examples/poml/weather_assistant.py` for a full working example with:

- Multiple tool definitions
- Conversation history
- Tool call handling
- Structured output
- VLM integration

## See Also

- [Client Guide](CLIENT_GUIDE.md) - API client examples
- [VLM Guide](VLM_GUIDE.md) - Vision-language model guide
- [API Reference](API_REFERENCE.md) - Full endpoint documentation

