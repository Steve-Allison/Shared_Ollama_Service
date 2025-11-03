# Model Storage Explanation

## Where Models Are Stored

Models are cached in a **Docker volume**, not in the project directory.

### Docker Volume Setup

```yaml
volumes:
  ollama_data:
    name: shared_ollama_data
    driver: local
```

This creates a Docker-managed volume that persists across container restarts.

### Benefits of Docker Volume Storage

1. **Persistence**: Models persist even if you `docker-compose down`
2. **Isolation**: Models stored separate from project code
3. **Performance**: Docker-optimized storage location
4. **Portability**: Easy to backup/restore volumes
5. **Cleanup**: Easy to remove with `docker-compose down -v`

### Inspecting the Volume

To see where Docker stores the models:

```bash
# List volumes
docker volume ls

# Inspect the volume
docker volume inspect shared_ollama_data

# Mount Point shows where Docker stores the data
# (typically in Docker's internal storage, not your home directory)
```

### Backup and Restore

**Backup:**
```bash
# Create backup
docker run --rm \
  -v shared_ollama_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/ollama-models-backup.tar.gz /data
```

**Restore:**
```bash
# Restore from backup
docker run --rm \
  -v shared_ollama_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/ollama-models-backup.tar.gz -C /
```

### Removing Models

**Keep models, remove container:**
```bash
docker-compose down  # Keeps volumes
```

**Remove everything:**
```bash
docker-compose down -v  # Also removes volumes and models!
```

### Model Size Reference

- **llama3.1:8b**: ~4.7 GB
- **mistral**: ~4.1 GB

**Total if all loaded**: ~8.8 GB in Docker volume

### Finding Models on Your System

**macOS:**
```bash
# Models stored in Docker's internal VM
# Path varies by Docker Desktop version
# Use docker volume inspect to find exact location
```

**Linux:**
```bash
# Models typically stored in:
# /var/lib/docker/volumes/shared_ollama_data/_data
```

### Alternative: Bind Mount to Project Directory

If you want models in your project directory instead:

```yaml
volumes:
  - ./models:/root/.ollama  # Change this line
```

**Pros:**
- Models visible in project directory
- Easy to backup with git
- Easy to delete manually

**Cons:**
- Large files in version control (not recommended)
- Slower on some systems
- Permissions issues possible

**Recommendation**: Keep using Docker volume (default) - it's optimized and cleaner.
