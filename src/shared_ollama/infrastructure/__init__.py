"""Infrastructure layer for Shared Ollama Service.

This package contains implementations of interfaces defined in the application
layer. These are framework-specific implementations that handle I/O, external
services, and infrastructure concerns.

The infrastructure layer:
    - Implements interfaces from application layer
    - Contains framework-specific code (FastAPI, httpx, requests, etc.)
    - Handles external service integrations
    - Manages file I/O, network, database, etc.
"""

from shared_ollama.infrastructure.adapters import (
    AnalyticsCollectorAdapter,
    AsyncOllamaClientAdapter,
    ImageCacheAdapter,
    ImageProcessorAdapter,
    MetricsCollectorAdapter,
    PerformanceCollectorAdapter,
    RequestLoggerAdapter,
)
from shared_ollama.infrastructure.health_checker import (
    check_ollama_health,
    check_ollama_health_simple,
)

__all__ = [
    "AsyncOllamaClientAdapter",
    "RequestLoggerAdapter",
    "MetricsCollectorAdapter",
    "ImageProcessorAdapter",
    "ImageCacheAdapter",
    "AnalyticsCollectorAdapter",
    "PerformanceCollectorAdapter",
    "check_ollama_health",
    "check_ollama_health_simple",
]

