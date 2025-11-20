"""Application layer for Shared Ollama Service.

This package contains use cases (application services) that orchestrate
domain logic. It depends only on the domain layer and defines interfaces
(protocols) for infrastructure dependencies.

The application layer:
    - Coordinates domain logic
    - Orchestrates workflows
    - Contains application-specific rules
    - Depends only on domain and interfaces (not implementations)
"""

from shared_ollama.application.interfaces import (
    AnalyticsCollectorInterface,
    ImageCacheInterface,
    ImageProcessorInterface,
    MetricsCollectorInterface,
    OllamaClientInterface,
    PerformanceCollectorInterface,
    RequestLoggerInterface,
)
from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)

__all__ = [
    "AnalyticsCollectorInterface",
    "ChatUseCase",
    "GenerateUseCase",
    "ImageCacheInterface",
    "ImageProcessorInterface",
    "ListModelsUseCase",
    "MetricsCollectorInterface",
    "OllamaClientInterface",
    "PerformanceCollectorInterface",
    "RequestLoggerInterface",
]
