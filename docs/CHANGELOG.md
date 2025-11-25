# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with pytest
- Async/await support (`AsyncSharedOllamaClient`)
- Monitoring and metrics collection
- Enhanced resilience (circuit breaker, exponential backoff)
- Type stubs for all modules
- CI/CD workflows (GitHub Actions)
- OpenAPI/Swagger API documentation
- Enhanced analytics with project-level tracking
- Python 3.13+ patterns (match/case, generic type parameters)

### Changed
- Modernized codebase to use Python 3.13+ patterns
- Improved type safety throughout
- Enhanced error handling with specific exceptions
- Better code organization and structure
- Upgraded dependency baseline: FastAPI 0.122.0, Pillow 12.0.0, cachetools 6.2.2, tenacity 9.1.2, psutil 7.1.3, pytest 9.0.1, pytest-asyncio 1.3.0, pytest-cov 7.0.0, ruff 0.14.6
- Added `SHARED_OLLAMA_FORCE_PROFILE` / `SHARED_OLLAMA_FORCE_RAM_GB` overrides for deterministic CI/test profiles

### Fixed
- Bare `except:` clauses replaced with specific exception handling
- Generic tuple return types specified
- Import organization improved

## [0.1.0] - 2025-11-05

### Added
- Initial release
- Shared Ollama client library
- Native macOS installation support
- Model management scripts
- Health check utilities
- Service discovery and configuration helpers

