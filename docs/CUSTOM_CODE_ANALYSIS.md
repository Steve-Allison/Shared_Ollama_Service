# Custom Code Analysis & Recommendations

## Overview
This document analyzes custom code implementations and recommends standard packages or native features that could replace them.

## ✅ Implementation Status

**All recommendations have been implemented!**

- ✅ Replaced watchdog.sh with gunicorn
- ✅ Replaced subprocess.Popen with asyncio.create_subprocess_exec
- ✅ Added psutil for process monitoring
- ✅ Updated start.sh and shutdown.sh for gunicorn

## Issues Found

### 1. ⚠️ **Watchdog Script (Custom Bash)**
**Location**: `scripts/watchdog.sh`

**Current Implementation**: Custom bash script that monitors uvicorn process and restarts on crash.

**Issues**:
- Platform-specific (bash, Unix-only)
- No built-in process management features
- Manual PID tracking
- Limited error recovery

**Recommended Solutions**:

#### Option A: Use `supervisor` (Python-based, cross-platform)
```bash
pip install supervisor
```

**Benefits**:
- Cross-platform (Linux, macOS, Windows)
- Built-in auto-restart, logging, process groups
- Configuration-based (no custom scripts)
- Production-ready with 15+ years of use

**Configuration** (`supervisord.conf`):
```ini
[program:shared_ollama_api]
command=/path/to/.venv/bin/uvicorn shared_ollama.api.server:app --host 0.0.0.0 --port 8000
directory=/path/to/project
autostart=true
autorestart=true
startretries=10
stderr_logfile=/path/to/logs/api.error.log
stdout_logfile=/path/to/logs/api.log
```

#### Option B: Use `gunicorn` with uvicorn workers (Better ASGI server)
```bash
pip install gunicorn
```

**Benefits**:
- Native process management (multiple workers)
- Auto-restart on worker crash
- Better performance (worker pool)
- Production-grade ASGI server

**Usage**:
```bash
gunicorn shared_ollama.api.server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info
```

#### Option C: Use native OS process managers
- **macOS**: `launchd` (already referenced in code)
- **Linux**: `systemd` service
- **Windows**: Windows Service

**Benefits**:
- Native OS integration
- Auto-start on boot
- Better resource management
- OS-level monitoring

---

### 2. ⚠️ **subprocess.Popen in Async Context**
**Location**: `src/shared_ollama/core/ollama_manager.py:364`

**Current Implementation**:
```python
self.process = subprocess.Popen(
    [ollama_path, "serve"],
    stdout=log_f,
    stderr=err_f,
    env=env,
    start_new_session=True,
)
```

**Issue**: Using synchronous `subprocess.Popen` in async function. This blocks the event loop.

**Recommended**: Use `asyncio.create_subprocess_exec`

**Benefits**:
- Non-blocking async subprocess creation
- Better integration with asyncio event loop
- Proper async/await support
- Better error handling

**Implementation**:
```python
self.process = await asyncio.create_subprocess_exec(
    ollama_path, "serve",
    stdout=log_f,
    stderr=err_f,
    env=env,
    start_new_session=True,
)
```

**Note**: Requires refactoring to handle async process management (wait, poll, etc.)

---

### 3. ✅ **Retry Logic (Good - Using tenacity)**
**Location**: `src/shared_ollama/core/resilience.py`

**Current Implementation**: Wraps `tenacity` library correctly.

**Status**: ✅ **GOOD** - Using proper package (`tenacity`) with custom wrapper for configuration.

**Recommendation**: Keep as-is. The wrapper provides good abstraction.

---

### 4. ⚠️ **Process Monitoring (Custom Bash)**
**Location**: `scripts/watchdog.sh`

**Current Implementation**: Custom bash script using `ps` and `curl` for health checks.

**Recommended**: Use `psutil` Python library

**Benefits**:
- Cross-platform process management
- Better process information (CPU, memory, threads)
- More reliable than parsing `ps` output
- Python-native (no shell scripts)

**Example**:
```python
import psutil

def is_process_running(pid: int) -> bool:
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()

def get_process_info(pid: int) -> dict:
    proc = psutil.Process(pid)
    return {
        "cpu_percent": proc.cpu_percent(),
        "memory_mb": proc.memory_info().rss / 1024 / 1024,
        "status": proc.status(),
    }
```

---

### 5. ✅ **Error Handling (Good - Using FastAPI)**
**Location**: `src/shared_ollama/api/middleware.py`

**Current Implementation**: Using FastAPI's built-in exception handlers correctly.

**Status**: ✅ **GOOD** - Using framework features properly.

---

### 6. ✅ **Queue Management (Good - Using asyncio)**
**Location**: `src/shared_ollama/core/queue.py`

**Current Implementation**: Using `asyncio.Semaphore` and `asyncio.Queue` correctly.

**Status**: ✅ **GOOD** - Using native asyncio primitives.

---

## Priority Recommendations

### High Priority

1. **Replace watchdog.sh with supervisor or gunicorn**
   - **Impact**: Production reliability, cross-platform support
   - **Effort**: Medium (configuration + testing)
   - **Recommendation**: Use `gunicorn` with uvicorn workers (simplest, best performance)

2. **Replace subprocess.Popen with asyncio.create_subprocess_exec**
   - **Impact**: Better async performance, no event loop blocking
   - **Effort**: Medium (refactor process management)
   - **Recommendation**: Use `asyncio.create_subprocess_exec` + `psutil` for monitoring

### Medium Priority

3. **Add psutil for process monitoring**
   - **Impact**: Better process information, cross-platform
   - **Effort**: Low (add dependency, update code)
   - **Recommendation**: Add `psutil` to requirements, update watchdog/manager

### Low Priority (Keep as-is)

- Retry logic (tenacity wrapper) ✅
- Error handling (FastAPI) ✅
- Queue management (asyncio) ✅

---

## Implementation Plan

### Phase 1: Replace Watchdog with Gunicorn
1. Add `gunicorn` to `pyproject.toml`
2. Update `start.sh` to use gunicorn instead of uvicorn directly
3. Configure worker count based on CPU cores
4. Remove `watchdog.sh` script
5. Update documentation

### Phase 2: Async Subprocess Management
1. Refactor `OllamaManager.start()` to use `asyncio.create_subprocess_exec`
2. Update process monitoring to use async methods
3. Add `psutil` for better process information
4. Update tests

### Phase 3: Optional - Supervisor Integration
1. Add supervisor configuration file
2. Document supervisor setup for production
3. Keep gunicorn as default, supervisor as alternative

---

## Dependencies to Add

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "gunicorn>=23.0.0",  # Production ASGI server with process management
    "psutil>=5.9.0",     # Cross-platform process management
]

[project.optional-dependencies]
production = [
    "supervisor>=4.2.0",  # Alternative process manager
]
```

---

## Summary

**Custom Code to Replace**:
1. ✅ `watchdog.sh` → `gunicorn` (recommended) or `supervisor`
2. ✅ `subprocess.Popen` → `asyncio.create_subprocess_exec`
3. ✅ Process monitoring → `psutil`

**Custom Code to Keep**:
- ✅ Retry logic wrapper (uses tenacity correctly)
- ✅ Error handling (uses FastAPI correctly)
- ✅ Queue management (uses asyncio correctly)

**Estimated Impact**:
- **Reliability**: ⬆️⬆️⬆️ (Better process management)
- **Maintainability**: ⬆️⬆️ (Less custom code)
- **Performance**: ⬆️⬆️ (Better async, worker pool)
- **Cross-platform**: ⬆️⬆️⬆️ (Python-native solutions)

