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

from shared_ollama.application.batch_use_cases import (
    BatchChatUseCase,
    BatchVLMUseCase,
)
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
from shared_ollama.application.vlm_use_cases import VLMUseCase

__all__ = [
    "AnalyticsCollectorInterface",
    "BatchChatUseCase",
    "BatchVLMUseCase",
    "ChatUseCase",
    "GenerateUseCase",
    "ImageCacheInterface",
    "ImageProcessorInterface",
    "ListModelsUseCase",
    "MetricsCollectorInterface",
    "OllamaClientInterface",
    "PerformanceCollectorInterface",
    "RequestLoggerInterface",
    "VLMUseCase",
]
