# Migration Guide: Consolidating to Shared Ollama Service

This guide helps you migrate from individual Ollama instances to the centralized shared service.

## Overview

Instead of running separate Ollama instances in each project, we'll use a single shared service on port `11434`.

## Benefits

- **Single Source of Truth**: One instance to manage
- **Resource Efficiency**: No duplicate services
- **Consistent Models**: All projects use same models
- **Easier Updates**: Update once, use everywhere
- **Better Monitoring**: Single health check point

## Pre-Migration Checklist

- [ ] Backup existing Ollama data
- [ ] Note down currently loaded models
- [ ] Review project-specific configurations
- [ ] Check if any custom Docker images are used

## Migration Steps

### Step 1: Setup Shared Service

```bash
cd Shared_Ollama_Service
./scripts/setup.sh
```

This will:
1. Start Docker Compose with Ollama service
2. Pull all required models
3. Run health checks
4. Verify everything is working

### Step 2: Update Knowledge Machine

**File**: `Knowledge_Machine/config/main.py`

**Before:**
```python
class OllamaConfig(BaseSettings):
    default_model: str = "llama3.1:8b"
    # ... local config
```

**After:**
```python
class OllamaConfig(BaseSettings):
    default_model: str = "qwen2.5vl:7b"
    base_url: str = "http://localhost:11434"  # Add this
```

**Update docker-compose** (if using):
```yaml
# Remove or comment out Ollama service
# ollama:
#   image: ollama/ollama:latest
#   ...

# Or keep but point to shared service
environment:
  - OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3: Update Course Intelligence Compiler

**File**: `Course_Intelligence_Compiler/config/rag_config.yaml`

**Before:**
```yaml
generation:
  ollama:
    base_url: "http://localhost:11434"  # Check current
    model: "mistral"  # or whatever model you're currently using
```

**After:**
```yaml
generation:
  ollama:
    base_url: "http://localhost:11434"  # Already correct
    model: "qwen2.5vl:7b"  # or "qwen2.5:14b" for alternative model
```

**Optional**: Update client code to use shared client:

```python
# Instead of Course_Intelligence_Compiler/src/common/llm/ollama_client.py
# Use Shared_Ollama_Service/shared_ollama_client.py

from shared_ollama_client import SharedOllamaClient

client = SharedOllamaClient(
    OllamaConfig(base_url="http://localhost:11434")
)
```

### Step 4: Update Story Machine

**File**: `Story_Machine/src/story_machine/core/config.py`

**Before:**
```python
class OllamaConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="mistral")
```

**After:**
```python
class OllamaConfig(BaseModel):
    base_url: str = Field(
        default="http://localhost:11434"  # Already correct!
    )
    model: str = Field(
        default="qwen2.5vl:7b"  # Or use qwen2.5:14b if needed
    )
```

**Update docker-compose.yml**:

**Before:**
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    # ... rest of config
```

**After:**
```yaml
# Remove local Ollama service
# Use shared service instead
# Just update environment variables to point to shared service
```

### Step 5: Stop Individual Ollama Services

```bash
# Stop in each project
cd Knowledge_Machine
docker-compose down  # Don't delete volumes yet!

cd Course_Intelligence_Compiler
docker-compose down

cd Story_Machine
docker-compose down
```

### Step 6: Test Each Project

#### Test Knowledge Machine

```bash
cd Knowledge_Machine

# Run tests
pytest tests/integration/test_rag_integration.py -v

# Or quick manual test
python -c "
from infrastructure.services.ollama_client import OllamaClient
client = OllamaClient()
response = client.generate('Hello!')
print(response.content)
"
```

#### Test Course Intelligence Compiler

```bash
cd Course_Intelligence_Compiler

# Run tests
pytest tests/common/llm/test_ollama_client.py -v

# Or quick manual test
python -c "
from src.common.llm.ollama_client import OllamaClient
client = OllamaClient(base_url='http://localhost:11434')
# Test generation
"
```

#### Test Story Machine

```bash
cd Story_Machine

# Run tests
pytest tests/ -k ollama -v

# Or check service
curl http://localhost:11434/api/tags
```

### Step 7: Cleanup (Optional)

After verifying everything works:

```bash
# Remove old Ollama data (optional)
# Only do this if you're sure everything is working!

# Knowledge Machine
cd Knowledge_Machine
docker-compose down -v  # Removes volumes

# Story Machine
cd Story_Machine
docker-compose down -v

# Keep Course_Intelligence_Compiler volumes if needed
```

## Rollback Plan

If something goes wrong:

```bash
# Stop shared service
cd Shared_Ollama_Service
docker-compose down

# Restart individual services
cd Knowledge_Machine
docker-compose up -d ollama

# Repeat for other projects
```

## Troubleshooting

### Issue: Models not found

**Solution:**
```bash
cd Shared_Ollama_Service
docker-compose exec ollama ollama pull qwen2.5vl:7b
docker-compose exec ollama ollama pull qwen2.5:14b
```

### Issue: Port 11434 already in use

**Solution:**
```bash
# Find what's using the port
lsof -i :11434

# Kill it
kill $(lsof -t -i:11434)

# Or use different port
# Update docker-compose.yml: "11435:11434"
# Update all project configs to use port 11435
```

### Issue: Connection refused

**Solution:**
```bash
# Check if service is running
docker ps | grep ollama

# Check logs
cd Shared_Ollama_Service
docker-compose logs ollama

# Restart service
docker-compose restart ollama
```

### Issue: Different model versions

Some projects might expect specific model versions (e.g., `qwen2.5vl:7b` vs `qwen2.5:14b`).

**Solution:**
1. Pull both versions
2. Update project configs to use the shared version
3. Or create aliases in Ollama:

```bash
docker-compose exec ollama ollama create qwen2.5vl:7b -f Modelfile
```

## Verification Checklist

After migration, verify:

- [ ] Shared Ollama service is running on port 11434
- [ ] All required models are available
- [ ] Health check passes: `./scripts/health_check.sh`
- [ ] Knowledge Machine tests pass
- [ ] Course Intelligence Compiler tests pass
- [ ] Story Machine tests pass
- [ ] No errors in project logs
- [ ] Model generation works in all projects

## Performance Comparison

### Before (Separate Instances)

- **Memory**: 3 instances × ~5GB = 15GB RAM
- **Startup**: Each instance loads models separately
- **Updates**: Need to update each instance

### After (Shared Service)

- **Memory**: 1 instance × ~8GB = 8GB RAM (savings!)
- **Startup**: Single model load
- **Updates**: Update once, all projects benefit

## Next Steps

1. Monitor performance and adjust resource limits
2. Consider adding more models if needed
3. Set up monitoring and alerts
4. Document any project-specific model preferences

## Support

If you encounter issues:

1. Check logs: `docker-compose logs ollama`
2. Run health check: `./scripts/health_check.sh`
3. Verify models: `docker-compose exec ollama ollama list`
4. Test API: `curl http://localhost:11434/api/tags`

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- Shared Ollama Service README: `Shared_Ollama_Service/README.md`
