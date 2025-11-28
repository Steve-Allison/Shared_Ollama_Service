# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Shared Ollama Service.

## Quick Troubleshooting Checklist

1. Capture `X-Shared-Ollama-Request-Id` and search it in `logs/api.log`.
2. `prompt_too_large` → shrink payload; call `/api/v1/system/model-profile` for live limits.
3. `queue_full` or `request_timeout` → respect `Retry-After`, stagger retries, and consider lowering client-side concurrency.
4. If issues persist, capture logs plus request metadata and open a ticket.

## Common Error Codes

### `prompt_too_large` (400)

**Cause**: Prompt history exceeded ~4,096 tokens.

**Solution**:
- Trim prompt history before sending
- Summarize older context instead of including raw history
- Call `/api/v1/system/model-profile` for current limits

### `request_too_large` (413)

**Cause**: JSON body exceeded **1.5 MiB**.

**Solution**:
- Compress or split large images
- Reduce image resolution before encoding
- Use image compression (enabled by default)

### `queue_full` (503)

**Cause**: Shared queue saturated.

**Solution**:
- Honor `Retry-After` header and stagger retries
- Reduce client-side concurrency
- Wait for queue to clear before retrying

### `request_timeout` (503)

**Cause**: Request waited 120s (text) / 150s (VLM) for a slot.

**Solution**:
- Reduce prompt size
- Lower concurrent requests
- Check queue status: `curl http://0.0.0.0:8000/api/v1/queue/stats`

## Service Not Starting

### Check Service Status

```bash
# Quick status check
./scripts/core/status.sh

# Comprehensive health check
./scripts/diagnostics/health_check.sh

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if REST API is running
curl http://0.0.0.0:8000/api/v1/health
```

### Common Startup Issues

**Port Already in Use**:
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process if needed
kill -9 <PID>
```

**Ollama Not Found**:
```bash
# Verify Ollama is installed
which ollama

# Install if missing
./scripts/install_native.sh
```

**Permission Issues**:
```bash
# Ensure scripts are executable
chmod +x scripts/**/*.sh
```

## Connection Issues

### Cannot Connect from Remote Client

1. **Check Firewall**: Ensure port 8000 is allowed
2. **Verify Network**: Both machines on same network
3. **Test Connectivity**: `curl http://<server-ip>:8000/api/v1/health`

### CORS Issues in Browser

- Ensure CORS is enabled (default in development)
- Check browser console for specific error
- Verify API base URL is correct

## Performance Issues

### Slow Response Times

1. **Check Model Warm-up**: Models may need to load
   ```bash
   ./scripts/warmup_models.sh
   ```

2. **Check Queue Status**: High queue depth indicates overload
   ```bash
   curl http://0.0.0.0:8000/api/v1/queue/stats
   ```

3. **Review Logs**: Check for errors or warnings
   ```bash
   tail -f logs/api.log
   tail -f logs/api.error.log
   ```

### High Memory Usage

- Check active models: `curl http://0.0.0.0:8000/api/v1/models`
- Reduce `OLLAMA_NUM_PARALLEL` if needed
- Adjust `RAG_RESERVE_GB` environment variable

## Log Analysis

### View Logs

```bash
# REST API logs
tail -f logs/api.log

# Error logs
tail -f logs/api.error.log

# Structured request log (JSON lines)
tail -f logs/requests.jsonl

# Performance logs
tail -f logs/performance.jsonl
```

### Search Logs by Request ID

```bash
# Find specific request
grep "request-id-here" logs/api.log

# Find errors
grep -i error logs/api.log
```

## Model Issues

### Model Not Found

1. **Check Available Models**:
   ```bash
   curl http://0.0.0.0:8000/api/v1/models
   ```

2. **Pull Missing Models**:
   ```bash
   ollama pull qwen3:14b-q4_K_M
   ```

3. **Pre-download All Models**:
   ```bash
   ./scripts/preload_models.sh
   ```

### Model Loading Errors

- Check available RAM
- Verify model files: `ls ~/.ollama/models`
- Review Ollama logs: `tail -f logs/ollama.log`

## Request Validation Errors

### Invalid Image Format

- Ensure images are base64-encoded data URLs
- Format: `data:image/jpeg;base64,<data>`
- Check image size (max 10MB recommended)

### Invalid Message Format

- Verify message structure matches API format
- Check role values: `system`, `user`, `assistant`
- Ensure content is a string or array (for multimodal)

## Getting Help

If issues persist:

1. **Collect Information**:
   - Request ID from `X-Shared-Ollama-Request-Id` header
   - Relevant log entries
   - Request payload (sanitized)
   - Service status output

2. **Check Documentation**:
   - [Stability Plan](STABILITY_PLAN.md) for known issues
   - [API Reference](API_REFERENCE.md) for endpoint details
   - [Client Usage Guidelines](../README.md#client-usage-guidelines-rag--chat)

3. **Open an Issue**: Include collected information and steps to reproduce

## See Also

- [Monitoring Guide](MONITORING.md) - Monitoring and observability
- [Resource Management](RESOURCE_MANAGEMENT.md) - Memory and performance tuning
- [Operations Guide](OPERATIONS.md) - Service operations and maintenance

