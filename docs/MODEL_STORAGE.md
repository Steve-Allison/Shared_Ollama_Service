# Model Storage Explanation

## Where Models Are Stored

Models are cached in your home directory: **`~/.ollama/models`** (native installation).

### Native Storage Location

**macOS:**
```
~/.ollama/models/
```

This directory is automatically created by Ollama when you first pull a model.

### Benefits of Native Storage

1. **Persistence**: Models persist across system restarts
2. **Accessibility**: Direct access to model files
3. **Performance**: No container overhead, fastest access
4. **Simplicity**: Native filesystem storage
5. **Backup**: Easy to backup/restore with standard tools

### Inspecting Models

To see where models are stored:

```bash
# List models
ollama list

# Check storage location
ls -lh ~/.ollama/models/

# Get size of models directory
du -sh ~/.ollama/models/
```

### Backup and Restore

**Backup:**
```bash
# Create backup
tar czf ~/ollama-models-backup.tar.gz ~/.ollama/models/
```

**Restore:**
```bash
# Restore from backup
tar xzf ~/ollama-models-backup.tar.gz -C ~
```

### Removing Models

**Remove a specific model:**
```bash
ollama rm model_name
```

**Remove all models:**
```bash
# Remove all models (be careful!)
rm -rf ~/.ollama/models/*
```

### Model Size Reference

- **llava:13b**: ~8 GB
- **qwen2.5:14b**: ~9 GB

**Total if all loaded**: ~17 GB in `~/.ollama/models/`

### Finding Models on Your System

**macOS:**
```bash
# Models stored in:
~/.ollama/models/

# View models
ls -lh ~/.ollama/models/

# Check disk usage
du -sh ~/.ollama/
```

**Check Model Information:**
```bash
# List installed models with sizes
ollama list

# Show model details
ollama show llava:13b
```

### Disk Space Management

**Check available space:**
```bash
df -h ~
```

**Clean up unused models:**
```bash
# Remove unused models
ollama rm model_name

# Models are stored as GGUF files
# You can manually delete files from ~/.ollama/models/ if needed
```

**Recommendation**: Keep models you use regularly. Models can be re-downloaded with `ollama pull` if needed.
