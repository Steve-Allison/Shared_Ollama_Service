# Project Structure

This document describes the organization of the Shared Ollama Service project, following Python best practices.

## 📁 Directory Structure

```
Shared_Ollama_Service/
├── .github/                    # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml             # CI pipeline
│       └── release.yml        # Release automation
├── docs/                       # Documentation
│   ├── README.md              # Documentation index
│   ├── CHANGELOG.md           # Version history
│   ├── API_REFERENCE.md       # Complete API documentation
│   ├── INTEGRATION_GUIDE.md   # Integration instructions
│   ├── MIGRATION_GUIDE.md     # Migration from individual instances
│   ├── MODEL_STORAGE.md       # Model storage information
│   ├── openapi.yaml           # OpenAPI 3.1.0 specification
│   ├── IMPLEMENTED_ENHANCEMENTS.md  # Enhancement details
│   ├── PYTHON_313_IMPROVEMENTS.md   # Python 3.13+ patterns
│   └── archive/               # Archived documentation
├── examples/                   # Usage examples
│   └── quick_start.py         # Quick start example
├── scripts/                    # Utility scripts
│   ├── ci_check.sh            # CI/CD health check
│   ├── health_check.sh        # Service health check
│   ├── install_native.sh      # Native installation
│   ├── preload_models.sh      # Pre-download models
│   ├── setup_launchd.sh       # macOS Launch Agent setup
│   ├── status.sh              # Quick status check
│   ├── verify_setup.sh        # Setup verification
│   ├── view_analytics.py      # Analytics dashboard
│   └── warmup_models.sh       # Model warm-up
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest configuration
│   ├── test_client.py         # Client tests
│   └── test_utils.py          # Utility tests
├── .gitignore                  # Git ignore rules
├── .pre-commit-config.yaml     # Pre-commit hooks
├── MANIFEST.in                 # Package manifest
├── Makefile                    # Development commands
├── README.md                   # Main project README
├── env.example                 # Environment variables template
├── pyproject.toml              # Project configuration (PEP 621)
├── requirements.txt            # Python dependencies
│
├── # Core library modules
├── shared_ollama_client.py     # Main synchronous client
├── shared_ollama_client.pyi    # Type stubs for main client
├── shared_ollama_client_async.py  # Async client
├── shared_ollama_client_async.pyi # Type stubs for async client
├── utils.py                    # Utility functions
├── utils.pyi                   # Type stubs for utils
├── monitoring.py               # Metrics collection
├── monitoring.pyi              # Type stubs for monitoring
├── resilience.py               # Resilience features
├── resilience.pyi              # Type stubs for resilience
├── analytics.py                # Enhanced analytics
└── analytics.pyi               # Type stubs for analytics
```

## 📦 Package Organization

### Source Code (Root Level)

The project uses a **flat package structure** which is appropriate for a library:

- **Core modules** at root level for easy import
- **Type stubs** (`.pyi` files) alongside each module
- **No nested package structure** - simplifies imports for consumers

### Why This Structure?

1. **Simple imports**: `from shared_ollama_client import SharedOllamaClient`
2. **Easy to use**: No complex package paths
3. **Type stubs**: Automatically detected by IDEs
4. **Standard practice**: Common for utility libraries

### Alternative Structure (Not Used)

For larger projects, a `src/` layout would be used:

```
src/
└── shared_ollama_service/
    ├── __init__.py
    ├── client.py
    ├── async_client.py
    └── ...
```

We use the flat structure because:
- Simpler for library consumers
- No need for package initialization
- Direct module imports

## 📚 Documentation Structure

### Active Documentation (`docs/`)

- **User-facing**: API Reference, Integration Guide, Migration Guide
- **Technical**: OpenAPI spec, Model Storage
- **Implementation**: Enhancement details, Python 3.13+ patterns

### Archived Documentation (`docs/archive/`)

- Historical planning documents
- Superseded enhancement plans
- Reference for project evolution

## 🧪 Testing Structure

```
tests/
├── __init__.py              # Test package marker
├── conftest.py              # Shared fixtures and configuration
├── test_client.py           # Client library tests
└── test_utils.py            # Utility function tests
```

**Test Organization**:
- One test file per module
- Shared fixtures in `conftest.py`
- Follows pytest best practices

## 🔧 Scripts Structure

```
scripts/
├── Installation & Setup
│   ├── install_native.sh
│   ├── setup_launchd.sh
│   └── verify_setup.sh
├── Service Management
│   ├── start.sh
│   ├── shutdown.sh
│   └── status.sh
├── Model Management
│   ├── preload_models.sh
│   └── warmup_models.sh
├── Monitoring & Health
│   ├── health_check.sh
│   └── ci_check.sh
└── Analytics
    └── view_analytics.py
```

## 📋 Configuration Files

### Root Level Configuration

- **`pyproject.toml`** - PEP 621 project metadata and tool configuration
- **`requirements.txt`** - Simple dependency list
- **`MANIFEST.in`** - Package distribution files
- **`.pre-commit-config.yaml`** - Pre-commit hooks
- **`Makefile`** - Development commands
- **`.gitignore`** - Version control exclusions

### Environment Configuration

- **`env.example`** - Template for environment variables
- **`.env`** - Local environment (gitignored)

## 🎯 Best Practices Followed

1. ✅ **Clear separation** of concerns (docs, tests, scripts, source)
2. ✅ **Type stubs** alongside source code
3. ✅ **Documentation** in dedicated directory
4. ✅ **Tests** mirror source structure
5. ✅ **Scripts** organized by purpose
6. ✅ **Configuration** at root level
7. ✅ **Build artifacts** excluded via `.gitignore`
8. ✅ **Examples** in dedicated directory

## 📝 File Naming Conventions

- **Python modules**: `snake_case.py`
- **Type stubs**: `snake_case.pyi`
- **Test files**: `test_*.py`
- **Scripts**: `snake_case.sh` or `snake_case.py`
- **Documentation**: `UPPERCASE.md` or `snake_case.md`
- **Config files**: `lowercase.ext` (e.g., `pyproject.toml`)

## 🔄 Migration Notes

If you need to reorganize in the future:

1. **To `src/` layout**: Create `src/shared_ollama_service/` and move modules
2. **To package structure**: Add `__init__.py` and create package hierarchy
3. **Update imports**: Modify `pyproject.toml` and update all imports

Current structure is optimal for a utility library with minimal dependencies.

