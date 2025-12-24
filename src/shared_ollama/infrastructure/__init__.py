"""Infrastructure layer for Shared Ollama Service.

Simplified infrastructure module containing only:
- Configuration (OllamaConfig)
- Health checker utilities
"""

from shared_ollama.infrastructure.config import OllamaConfig
from shared_ollama.infrastructure.health_checker import (
    check_ollama_health,
    check_ollama_health_simple,
)

__all__ = [
    "OllamaConfig",
    "check_ollama_health",
    "check_ollama_health_simple",
]
