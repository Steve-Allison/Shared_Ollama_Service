# Architecture

This document describes the architecture of the Shared Ollama Service, including system design, component structure, and request flow.

## Overview

The Shared Ollama Service follows **Clean Architecture** principles with strict layer separation and dependency inversion. The system provides both a REST API and Python client libraries for accessing Ollama models with centralized logging, metrics, and resilience features.

## Architecture Layers

The codebase is organized into four main layers following Clean Architecture:

### 1. Domain Layer (`src/shared_ollama/domain/`)

**Purpose**: Pure business logic with no external dependencies.

**Components**:
- **Entities** (`entities.py`):
  - `Model` - Enum of available models
  - `ModelInfo` - Model metadata
  - `GenerationRequest` - Domain entity for generation requests
  - `ChatRequest` - Domain entity for chat requests
  - `ChatMessage` - Domain entity for chat messages
  - `VLMRequest` - Vision-language model requests
  - `GenerationOptions` - Generation parameters with validation

- **Value Objects** (`value_objects.py`):
  - `ModelName` - Validated model name
  - `Prompt` - Validated prompt text
  - `SystemMessage` - System message text

- **Exceptions** (`exceptions.py`):
  - `DomainError` - Base domain exception
  - `InvalidModelError` - Invalid model name
  - `InvalidPromptError` - Invalid prompt
  - `InvalidRequestError` - Invalid request

**Key Characteristics**:
- No framework imports
- No I/O operations
- Pure Python dataclasses with `slots=True`
- Business rule validation in `__post_init__`
- Immutable entities (frozen dataclasses)

### 2. Application Layer (`src/shared_ollama/application/`)

**Purpose**: Orchestrates domain logic and coordinates workflows.

**Components**:
- **Interfaces** (`interfaces.py`):
  - `OllamaClientInterface` - Protocol for Ollama client implementations
  - `RequestLoggerInterface` - Protocol for request logging
  - `MetricsCollectorInterface` - Protocol for metrics collection

- **Use Cases** (`use_cases.py`, `batch_use_cases.py`, `vlm_use_cases.py`):
  - `GenerateUseCase` - Orchestrates text generation
  - `ChatUseCase` - Orchestrates chat completion
  - `ListModelsUseCase` - Orchestrates model listing
  - `BatchChatUseCase` - Batch chat processing
  - `BatchVLMUseCase` - Batch VLM processing
  - `VLMUseCase` - Vision-language model processing with image handling

**Key Characteristics**:
- Depends only on domain entities and interfaces
- No framework dependencies
- Handles logging and metrics via injected adapters
- Coordinates between domain and infrastructure

**VLMUseCase Dependencies**:

The VLMUseCase has additional dependencies compared to standard use cases due to image processing requirements:

*Required Dependencies:*

1. `client: OllamaClientInterface` - Async Ollama HTTP client for model inference
2. `logger: RequestLoggerInterface` - Structured logging for request tracking
3. `metrics: MetricsCollectorInterface` - Request metrics and performance tracking
4. `image_processor: ImageProcessorInterface` - Image compression and format validation
5. `image_cache: ImageCacheInterface` - LRU cache for processed images (SHA-256 deduplication)

*Optional Dependencies:*
6. `analytics: AnalyticsCollectorInterface | None` - Project-based analytics and usage patterns
7. `performance: PerformanceCollectorInterface | None` - Detailed performance metrics and token rates

**Dependency Injection Pattern:**

```python
# In api/dependencies.py
def get_vlm_use_case(
    client_adapter: Annotated[AsyncOllamaClientAdapter, Depends(get_client_adapter)],
    logger_adapter: Annotated[RequestLoggerAdapter, Depends(get_logger_adapter)],
    metrics_adapter: Annotated[MetricsCollectorAdapter, Depends(get_metrics_adapter)],
    image_processor_adapter: Annotated[ImageProcessorAdapter, Depends(get_image_processor)],
    image_cache_adapter: Annotated[ImageCacheAdapter, Depends(get_image_cache)],
) -> VLMUseCase:
    """Get VLMUseCase instance with all required dependencies."""
    return VLMUseCase(
        client=client_adapter,
        logger=logger_adapter,
        metrics=metrics_adapter,
        image_processor=image_processor_adapter,
        image_cache=image_cache_adapter,
        analytics=_analytics_adapter,  # Optional global
        performance=_performance_adapter,  # Optional global
    )
```

**Image Processing Pipeline:**

1. **Validation**: Verify base64 data URL format and extract image bytes
2. **Cache Lookup**: Check LRU cache using SHA-256 hash of original image
3. **Compression**: If not cached, compress to target format (JPEG/PNG/WebP)
4. **Cache Storage**: Store compressed image with metadata (original size, compressed size, format)
5. **Metrics Tracking**: Record compression savings, cache hit rate, images processed

The image cache uses an LRU eviction policy with:

- 1-hour TTL per entry
- 1GB maximum total cache size
- SHA-256-based deduplication
- Thread-safe operations

### 3. Interface Adapters (`src/shared_ollama/api/`)

**Purpose**: Adapts between external interfaces (HTTP) and application layer.

**Components**:
- **API Models** (`models.py`):
  - Pydantic models for request/response validation
  - FastAPI-specific models
  - Request/response DTOs

- **Mappers** (`mappers.py`):
  - `api_to_domain_generation_request()` - Converts API model to domain entity
  - `api_to_domain_chat_request()` - Converts API model to domain entity
  - `domain_to_api_model_info()` - Converts domain entity to API model

- **Dependencies** (`dependencies.py`):
  - FastAPI dependency injection functions
  - Removes global state
  - Provides use case instances via DI

- **Routes** (`routes/*.py`):
  - Modular route handlers
  - `generation.py` - Text generation endpoints
  - `chat.py` - Chat completion endpoints
  - `vlm.py` - Vision-language model endpoints
  - `batch.py` - Batch processing endpoints
  - `system.py` - System endpoints (health, models, metrics)

- **Server** (`server.py`):
  - FastAPI application setup
  - Middleware configuration
  - Router registration

**Key Characteristics**:
- Depends on application layer (use cases)
- No direct domain access (via use cases)
- Framework-specific code (FastAPI)
- Thin controllers with minimal logic

### 4. Infrastructure Layer (`src/shared_ollama/infrastructure/`)

**Purpose**: Framework and external service implementations.

**Components**:
- **Adapters** (`adapters.py`):
  - `AsyncOllamaClientAdapter` - Wraps `AsyncSharedOllamaClient` to implement `OllamaClientInterface`
  - `RequestLoggerAdapter` - Wraps structured logging to implement `RequestLoggerInterface`
  - `MetricsCollectorAdapter` - Wraps metrics collector to implement `MetricsCollectorInterface`

- **Image Processing** (`image_processing.py`, `image_cache.py`):
  - Image compression and optimization for VLM
  - LRU cache for processed images

- **Health Checker** (`health_checker.py`):
  - Service health verification

**Key Characteristics**:
- Implements interfaces from application layer
- Framework-specific code (httpx, requests, etc.)
- External service integrations
- File I/O, network, database operations

## Dependency Flow

```
Framework (FastAPI)
    ↓
Interface Adapters (API Controllers)
    ↓
Application Layer (Use Cases)
    ↓
Domain Layer (Entities, Value Objects)
    ↑
Infrastructure (Adapters implement Application Interfaces)
```

**Dependency Rule**: Dependencies always point inward. Inner layers never depend on outer layers.

## System Components

### Core Modules (`src/shared_ollama/core/`)

- **Configuration** (`config.py`):
  - Centralized configuration management using pydantic-settings
  - Environment variable loading and validation
  - Type-safe configuration access

- **Ollama Manager** (`ollama_manager.py`):
  - Ollama process lifecycle management
  - Auto-detection of system optimizations
  - Process monitoring and health checks

- **Request Queue** (`queue.py`):
  - Async request queue with configurable concurrency
  - Separate queues for chat and VLM requests
  - Queue statistics and monitoring

- **Resilience** (`resilience.py`):
  - Circuit breaker pattern
  - Retry strategies with exponential backoff
  - Self-healing capabilities

- **Utils** (`utils.py`):
  - Service discovery and health checks
  - Project root detection
  - Client path resolution

### Client Libraries (`src/shared_ollama/client/`)

- **Sync Client** (`sync.py`):
  - Synchronous HTTP client using `requests`
  - Connection pooling and retries
  - Structured logging integration

- **Async Client** (`async_client.py`):
  - Asynchronous HTTP client using `httpx`
  - Connection pooling and concurrency control
  - Streaming support
  - Tool calling and format support

### Telemetry (`src/shared_ollama/telemetry/`)

- **Metrics** (`metrics.py`):
  - Request metrics collection
  - Model usage statistics
  - Performance tracking

- **Analytics** (`analytics.py`):
  - Project-based analytics
  - Usage patterns and trends

- **Performance** (`performance.py`):
  - Detailed performance metrics
  - Model load times and warm start tracking

- **Structured Logging** (`structured_logging.py`):
  - JSON-formatted request logs
  - Event tracking and correlation

## Request Flow

### Via REST API (Recommended)

```
┌────────────────────┐    HTTP/REST
│  Client Project     │ ────────────────────────────────────────┐
│  (Any Language)     │                                           │
└────────────────────┘                                           ▼
                                                          ┌──────────────────────┐
                                                          │  FastAPI REST API    │
                                                          │  (Port 8000, Async)  │
                                                          │  - Rate limiting     │
                                                          │  - Request tracking  │
                                                          │  - Structured logs   │
                                                          │  - Request queuing   │
                                                          └──────────────────────┘
                                                                  │  Uses
                                                                  ▼
                                                          ┌───────────────────────────┐
                                                          │  Use Cases (via DI)       │
                                                          │  - GenerateUseCase        │
                                                          │  - ChatUseCase           │
                                                          │  - VLMUseCase            │
                                                          └───────────────────────────┘
                                                                  │  Uses
                                                                  ▼
                                                          ┌───────────────────────────┐
                                                          │ AsyncSharedOllamaClient   │
                                                          │ (httpx.AsyncClient)       │
                                                          └───────────────────────────┘
                                                                  │  http(s) JSON
                                                                  ▼
                                                          ┌──────────────────────┐
                                                          │     Ollama API      │
                                                          │  (HTTP on :11434)   │
                                                          └──────────────────────┘
                                                                  │  GGUF model loads
                                                                  ▼
                                                          ┌──────────────────────┐
                                                          │   Model Runtime      │
                                                          │  (GPU/CPU via MPS)   │
                                                          └──────────────────────┘
```

### Via Direct Client Library (Python Only)

```
┌────────────────────┐    generate() / chat() / stream()
│  Client Project     │ ────────────────────────────────────────┐
│  (Python)           │                                           │
└────────────────────┘                                           ▼
                    configure client + ensure running       ┌───────────────────────────┐
                                                            │ shared_ollama.client.sync │
                                                            │ shared_ollama.client.async │
                                                            └───────────────────────────┘
                                                                  │  http(s) JSON
                                                                  ▼
                                                            ┌──────────────────────┐
                                                            │     Ollama API      │
                                                            │  (HTTP on :11434)   │
                                                            └──────────────────────┘
                                                                  │  GGUF model loads
                                                                  ▼
                                                            ┌──────────────────────┐
                                                            │   Model Runtime      │
                                                            │  (GPU/CPU via MPS)   │
                                                            └──────────────────────┘
```

## Key Design Decisions

### 1. Clean Architecture

- **Separation of Concerns**: Each layer has a single responsibility
- **Dependency Inversion**: Inner layers define interfaces, outer layers implement them
- **Testability**: Easy to test with protocol-based interfaces
- **Maintainability**: Clear boundaries make code easier to understand and modify

### 2. Dependency Injection

- **FastAPI Depends()**: All dependencies injected via FastAPI's dependency system
- **No Global State**: Removed global variables in favor of dependency injection
- **Protocol-Based**: Use Python protocols for interface definitions
- **Testability**: Easy to mock dependencies for testing

### 3. Configuration Management

- **Centralized**: Single source of truth for all configuration
- **Type-Safe**: Pydantic models with validation
- **Profile-Driven**: Auto-selects defaults from `config/model_profiles.yaml` with optional environment overrides
- **Validated**: All values validated at startup

### 4. Request Queuing

- **Separate Queues**: Different queues for chat and VLM requests
- **Configurable Concurrency**: Adjustable based on system resources
- **Timeout Handling**: Configurable timeouts for queue waits
- **Statistics**: Comprehensive queue metrics for monitoring

### 5. Image Processing

- **Compression**: Automatic image compression for VLM requests
- **Caching**: LRU cache for processed images
- **Format Support**: JPEG, PNG, WebP support
- **Size Limits**: Configurable maximum image dimensions and sizes

## Runtime Environments

| Layer        | Description                                             |
|--------------|---------------------------------------------------------|
| Dev Machines | Local `.venv`, manual `ollama serve`, scripts to manage |
| CI           | Headless checks via `scripts/ci_check.sh`, mocked tests |
| Production   | Long-running Ollama daemon, monitored via logs + metrics |

### Ollama Service Configuration

- `OLLAMA_METAL=1`, `OLLAMA_NUM_GPU=-1` for Apple Silicon acceleration
- `OLLAMA_MAX_RAM` dynamically computed by `calculate_memory_limit.sh`
- Keep-alive of 5 minutes ensures idle models unload to reclaim memory
- See [CONFIGURATION.md](CONFIGURATION.md) for all configuration options

## Operational Checklist

1. Start Ollama (`ollama serve` or `scripts/start.sh`)
2. Verify health (`scripts/status.sh`, `scripts/verify_setup.sh`)
3. Run clients/tests with `.venv` activated (`pip install -e ".[dev]" -c constraints.txt` recommended)
4. Monitor logs (`logs/ollama.log`, `ollama.error.log`) or forward to observability platform
5. Use `scripts/performance_report.py` to baseline latency after model upgrades

## See Also

- [Configuration Guide](CONFIGURATION.md) - Complete configuration reference
- [API Reference](API_REFERENCE.md) - API documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - How to integrate the service
- [Development Guide](DEVELOPMENT.md) - Development and testing guide
