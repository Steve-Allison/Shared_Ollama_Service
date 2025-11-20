# Integration Guide - Using Shared Ollama Service in Your Projects

This guide shows you how to integrate the Shared Ollama Service into your projects.

## Quick Integration (3 Steps)

### Step 1: Ensure Service is Running

```bash
# From Shared_Ollama_Service directory
./scripts/verify_setup.sh
```

Or check status:
```bash
./scripts/status.sh
```

### Step 2: Add Service URL to Your Project

Set environment variable (recommended):
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
```

Or add to your `.env` file:
```env
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3: Use the Client

#### Option A: Import from Shared Service (Recommended)

```python
import sys
from pathlib import Path

# Add Shared_Ollama_Service to path
sys.path.insert(0, str(Path.home() / "AI_Projects+Code" / "Shared_Ollama_Service"))

from shared_ollama import SharedOllamaClient

# Use it!
client = SharedOllamaClient()
response = client.generate("Hello, world!")
print(response.text)
```

#### Option B: Use Utils for Service Discovery

```python
from utils import get_ollama_base_url, ensure_service_running
from shared_ollama import OllamaConfig, SharedOllamaClient

# Automatically get URL from environment
base_url = get_ollama_base_url()
ensure_service_running(base_url)

client = SharedOllamaClient(OllamaConfig(base_url=base_url))
```

#### Option C: Direct HTTP API Calls

```python
import requests
import os

base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

response = requests.post(
    f"{base_url}/api/generate",
    json={
        "model": "qwen3-vl:8b-instruct-q4_K_M",
        "prompt": "Hello, world!",
        "stream": False
    }
)
result = response.json()
print(result["response"])
```

#### Option D: TypeScript / JavaScript (Node)

```ts
import fetch from "node-fetch";

const baseUrl = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";

async function generate(prompt: string) {
  const res = await fetch(`${baseUrl}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "qwen3-vl:8b-instruct-q4_K_M",
      prompt,
      stream: false,
    }),
  });

  if (!res.ok) {
    throw new Error(`Ollama error: ${res.status} ${await res.text()}`);
  }

  const { response } = (await res.json()) as { response: string };
  return response;
}

generate("Hello, world!").then(console.log).catch(console.error);
```

## Project-Specific Integration

### Knowledge Machine

**File**: `Knowledge_Machine/config/main.py`

```python
from pydantic_settings import BaseSettings

class OllamaConfig(BaseSettings):
    base_url: str = "http://localhost:11434"
    default_model: str = "qwen3-vl:8b-instruct-q4_K_M"

    class Config:
        env_prefix = "OLLAMA_"
```

**Usage in code**:
```python
from config.main import OllamaConfig
import sys
sys.path.insert(0, "/path/to/Shared_Ollama_Service")
from shared_ollama import SharedOllamaClient

config = OllamaConfig()
client = SharedOllamaClient(
    OllamaConfig(base_url=config.base_url, default_model=config.default_model)
)
```

### Course Intelligence Compiler

**File**: `Course_Intelligence_Compiler/config/rag_config.yaml`

```yaml
generation:
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen3-vl:8b-instruct-q4_K_M"  # or switch to "qwen3:30b" on high-memory hosts
    timeout: 120
```

**Python usage**:
```python
import yaml
from shared_ollama import OllamaConfig, SharedOllamaClient

with open("config/rag_config.yaml") as f:
    config = yaml.safe_load(f)

ollama_config = config["generation"]["ollama"]
client = SharedOllamaClient(
    OllamaConfig(
        base_url=ollama_config["base_url"],
        default_model=ollama_config["model"],
        timeout=ollama_config.get("timeout", 60)
    )
)
```

### Story Machine

**File**: `Story_Machine/src/story_machine/core/config.py`

```python
from pydantic import BaseModel, Field
import sys
from pathlib import Path

# Add Shared_Ollama_Service to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "Shared_Ollama_Service"))

from shared_ollama import OllamaConfig, SharedOllamaClient

class StoryConfig(BaseModel):
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen3-vl:8b-instruct-q4_K_M")

    def get_ollama_client(self):
        return SharedOllamaClient(
            OllamaConfig(
                base_url=self.ollama_base_url,
                default_model=self.ollama_model
            )
        )
```

## Testing Integration

### Pre-test Setup

Add to your `conftest.py` or test setup:

```python
import pytest
import sys
from pathlib import Path

# Add Shared_Ollama_Service to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Shared_Ollama_Service"))

from utils import ensure_service_running

@pytest.fixture(scope="session", autouse=True)
def ensure_ollama_service():
    """Ensure Ollama service is running before tests."""
    ensure_service_running()
```

### CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# GitHub Actions example
- name: Check Ollama Service
  run: |
    cd Shared_Ollama_Service
    ./scripts/ci_check.sh

- name: Run tests
  run: pytest
```

Or use the check script:
```bash
cd Shared_Ollama_Service
./scripts/ci_check.sh || echo "Service check skipped"
```

## Error Handling

### Graceful Degradation

```python
from shared_ollama import SharedOllamaClient
from utils import check_service_health

is_healthy, error = check_service_health()
if not is_healthy:
    # Fallback to alternative or skip feature
    print(f"Ollama unavailable: {error}")
    # Use alternative LLM or skip feature
else:
    client = SharedOllamaClient()
    response = client.generate("Hello!")
```

### Retry Logic

The client includes automatic retry on connection. For additional retries:

```python
from shared_ollama import SharedOllamaClient
import time

client = SharedOllamaClient(verify_on_init=False)  # Skip initial verification

def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.generate(prompt)
        except ConnectionError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Environment Variables

All projects can use these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for Ollama service |
| `OLLAMA_HOST` | `localhost` | Host (alternative to base_url) |
| `OLLAMA_PORT` | `11434` | Port (alternative to base_url) |
| `SHARED_OLLAMA_SERVICE_PATH` | Auto-detected | Path to Shared_Ollama_Service directory |

## Examples

See `examples/quick_start.py` for complete working examples.

## Troubleshooting

### Service Not Found

```bash
# Check if service is running
./scripts/status.sh

# Start service
./scripts/start.sh  # REST API manages Ollama internally
```

### Import Errors

```python
# Use utils to find the service directory
from utils import import_client

SharedOllamaClient = import_client()
client = SharedOllamaClient()
```

### Connection Refused

1. Check service is running: `./scripts/status.sh`
2. Verify URL is correct: `echo $OLLAMA_BASE_URL`
3. Check firewall/network settings
4. Restart service: `./scripts/shutdown.sh && ./scripts/start.sh`

## Next Steps

- See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migrating from individual instances
- Review the [Architecture Overview](ARCHITECTURE.md) for component responsibilities
- See [README.md](../README.md) for complete documentation
- Run examples: `python examples/quick_start.py`

