"""Core helpers for the Shared Ollama Service."""

from shared_ollama.core.ollama_manager import (
    OllamaManager,
    get_ollama_manager,
    initialize_ollama_manager,
)
from shared_ollama.core.utils import (
    check_service_health,
    ensure_service_running,
    get_allowed_models,
    get_client_path,
    get_default_text_model,
    get_default_vlm_model,
    get_model_profile_summary,
    get_ollama_base_url,
    get_project_root,
    import_client,
    is_model_allowed,
)
from shared_ollama.infrastructure.config import (
    OllamaConfig,
)

__all__ = [
    "OllamaConfig",
    "OllamaManager",
    "check_service_health",
    "ensure_service_running",
    "get_allowed_models",
    "get_client_path",
    "get_default_text_model",
    "get_default_vlm_model",
    "get_model_profile_summary",
    "get_ollama_base_url",
    "get_ollama_manager",
    "get_project_root",
    "import_client",
    "initialize_ollama_manager",
    "is_model_allowed",
]
