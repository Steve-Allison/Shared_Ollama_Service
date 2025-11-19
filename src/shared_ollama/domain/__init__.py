"""Domain layer for Shared Ollama Service.

This package contains pure domain models, value objects, and business rules
with no dependencies on frameworks, infrastructure, or external libraries.

The domain layer is the innermost layer and has no dependencies on outer layers.
"""

from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationRequest,
    Model,
    ModelInfo,
)
from shared_ollama.domain.exceptions import (
    DomainError,
    InvalidModelError,
    InvalidPromptError,
    InvalidRequestError,
)
from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage

__all__ = [
    # Entities
    "Model",
    "ModelInfo",
    "GenerationRequest",
    "ChatRequest",
    "ChatMessage",
    # Value Objects
    "ModelName",
    "Prompt",
    "SystemMessage",
    # Exceptions
    "DomainError",
    "InvalidModelError",
    "InvalidPromptError",
    "InvalidRequestError",
]
