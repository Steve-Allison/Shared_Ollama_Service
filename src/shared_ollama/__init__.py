"""Shared Ollama Service Python SDK - Pure Ollama Client Library."""

from shared_ollama.client import (
    AsyncOllamaConfig,
    AsyncSharedOllamaClient,
    GenerateOptions,
    GenerateResponse,
    Model,
    OllamaConfig,
    SharedOllamaClient,
)
from shared_ollama.core import (
    check_service_health,
    ensure_service_running,
    get_client_path,
    get_default_text_model,
    get_default_vlm_model,
    get_ollama_base_url,
    get_project_root,
    import_client,
    is_model_allowed,
    get_allowed_models,
    get_model_profile_summary,
)

__all__ = [
    "AsyncOllamaConfig",
    "AsyncSharedOllamaClient",
    "GenerateOptions",
    "GenerateResponse",
    "Model",
    "OllamaConfig",
    "SharedOllamaClient",
    "check_service_health",
    "ensure_service_running",
    "get_allowed_models",
    "get_client_path",
    "get_default_text_model",
    "get_default_vlm_model",
    "get_model_profile_summary",
    "get_ollama_base_url",
    "get_project_root",
    "import_client",
    "is_model_allowed",
]
