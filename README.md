# Shared Ollama Service

A centralized Ollama instance for all AI projects to reduce duplication and improve resource management.

## Overview

This service provides a single Ollama instance accessible on port `11434` that all projects can use:
- **Knowledge Machine**
- **Course Intelligence Compiler**  
- **Story Machine**

## Models Loaded

- **Primary**: `llava:13b` (13B parameters, vision model)
  - Vision-language model with multimodal capabilities
  - Best overall performance and reasoning
- **Secondary**: `qwen2.5:14b` (14.8B parameters)
  - Large language model with excellent reasoning
  - Good alternative for text-only tasks

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama installed locally OR use the Docker image

### Start the Service

```bash
# Start Ollama service
docker-compose up -d

# Or start with logs visible
docker-compose up
```

### Pull Models

```bash
# Pull primary model (llava:13b)
docker-compose exec ollama ollama pull llava:13b

# Pull secondary model (qwen2.5:14b)
docker-compose exec ollama ollama pull qwen2.5:14b
```

### Verify Installation

```bash
# Run health check
./scripts/health_check.sh

# Or manually check
curl http://localhost:11434/api/tags
```

## Usage in Projects

### Knowledge Machine

Update `Knowledge_Machine/config/main.py`:
```python
ollama_base_url = "http://localhost:11434"
default_model = "llava:13b"
```

### Course Intelligence Compiler

Update `Course_Intelligence_Compiler/config/rag_config.yaml`:
```yaml
generation:
  ollama:
    base_url: "http://localhost:11434"
    model: "llava:13b"  # or "qwen2.5:14b" for alternative model
```

### Story Machine

Update `Story_Machine/src/story_machine/core/config.py`:
```python
ollama:
    base_url: "http://localhost:11434"
    model: "llava:13b"  # or "qwen2.5:14b" for specific use cases
```

## Configuration

### Environment Variables

Create `.env` file:
```env
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
OLLAMA_KEEP_ALIVE=5m
OLLAMA_DEBUG=false
```

### Port Configuration

- **Default Port**: 11434 (standard Ollama port)
- **API Endpoint**: `http://localhost:11434/api/generate`
- **Tags Endpoint**: `http://localhost:11434/api/tags`

## Health Checks

### Automated Health Check

```bash
./scripts/health_check.sh
```

This checks:
- Ollama service is running
- All models are available
- API is responding

### Manual Testing

```bash
# List models
curl http://localhost:11434/api/tags

# Test generation
curl http://localhost:11434/api/generate -d '{
  "model": "llava:13b",
  "prompt": "Why is the sky blue?"
}'
```

## Model Management

### List Loaded Models

```bash
docker-compose exec ollama ollama list
```

### Remove a Model

```bash
docker-compose exec ollama ollama rm model_name
```

### Update Models

```bash
# Pull latest version
docker-compose exec ollama ollama pull llava:13b
```

## Performance Tuning

### Hardware Detection

The service auto-detects hardware:
- **Apple Silicon** (M1/M2/M3): Full Metal acceleration
- **NVIDIA GPU**: CUDA acceleration
- **CPU Only**: Fallback mode

### Resource Limits

Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4'
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs ollama

# Restart service
docker-compose restart ollama
```

### Models Not Found

```bash
# Pull models
docker-compose exec ollama ollama pull llava:13b

# Verify
docker-compose exec ollama ollama list
```

### Connection Refused

```bash
# Check if port is in use
lsof -i :11434

# Kill existing Ollama instance
kill $(lsof -t -i:11434)

# Restart
docker-compose up -d ollama
```

## Migration Guide

### From Individual Project Ollama Instances

1. **Stop existing instances**
   ```bash
   # Stop Ollama in each project
   cd Knowledge_Machine && docker-compose down
   cd Course_Intelligence_Compiler && docker-compose down
   cd Story_Machine && docker-compose down
   ```

2. **Start shared service**
   ```bash
   cd Shared_Ollama_Service
   docker-compose up -d
   ```

3. **Update project configurations** (see Usage section above)

4. **Test each project**
   ```bash
   # Test Knowledge Machine
   cd Knowledge_Machine && python -m pytest tests/integration/test_rag_integration.py
   
   # Test Course Intelligence Compiler
   cd Course_Intelligence_Compiler && python -m pytest tests/common/llm/test_ollama_client.py
   
   # Test Story Machine
   cd Story_Machine && pytest tests/
   ```

## Security

### Network Isolation

The Ollama service is **not exposed to the internet** by default. Only localhost connections are allowed.

### Model Access Control

Models are stored in Docker volume: `ollama_data:/root/.ollama`

## Monitoring

### Logs

```bash
# View logs
docker-compose logs -f ollama

# View last 100 lines
docker-compose logs --tail=100 ollama
```

### Metrics

Monitor with:
- **Health checks**: `scripts/health_check.sh`
- **Model status**: `curl http://localhost:11434/api/tags`
- **Resource usage**: `docker stats shared-ollama-service-ollama-1`

## Cost and Resource Management

### Memory Usage

- `llava:13b`: ~8 GB RAM
- `qwen2.5:14b`: ~9 GB RAM

**Recommendation**: Load only models you need.

### Performance

- **First request**: ~2-3 seconds (model loading)
- **Subsequent requests**: ~100-500ms (depends on prompt length)

## Contributing

When adding new models or modifying the service:

1. Update `docker-compose.yml`
2. Update this README
3. Test with all projects
4. Document model size and use cases

## License

MIT

## Support

For issues or questions:
- Check logs: `docker-compose logs ollama`
- Run health check: `./scripts/health_check.sh`
- Open issue in project repository
