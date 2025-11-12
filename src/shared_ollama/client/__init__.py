"""Client interfaces for the Shared Ollama Service."""

from shared_ollama.client.async_client import AsyncOllamaConfig, AsyncSharedOllamaClient
from shared_ollama.client.sync import (
    GenerateOptions,
    GenerateResponse,
    Model,
    OllamaConfig,
    SharedOllamaClient,
)

__all__ = [
    "AsyncOllamaConfig",
    "AsyncSharedOllamaClient",
    "GenerateOptions",
    "GenerateResponse",
    "Model",
    "OllamaConfig",
    "SharedOllamaClient",
]

