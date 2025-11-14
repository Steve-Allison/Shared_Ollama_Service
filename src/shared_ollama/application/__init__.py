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
    MetricsCollectorInterface,
    OllamaClientInterface,
    RequestLoggerInterface,
)
from shared_ollama.application.use_cases import (
    ChatUseCase,
    GenerateUseCase,
    ListModelsUseCase,
)

__all__ = [
    # Interfaces
    "OllamaClientInterface",
    "RequestLoggerInterface",
    "MetricsCollectorInterface",
    # Use Cases
    "GenerateUseCase",
    "ChatUseCase",
    "ListModelsUseCase",
]

