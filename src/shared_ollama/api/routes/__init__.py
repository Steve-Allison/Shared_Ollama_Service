"""API routes for the Shared Ollama Service.

Modular route definitions organized by functionality.
"""

from shared_ollama.api.routes.agents import router as agents_router
from shared_ollama.api.routes.batch import router as batch_router
from shared_ollama.api.routes.chat import router as chat_router
from shared_ollama.api.routes.embeddings import router as embeddings_router
from shared_ollama.api.routes.generation import router as generation_router
from shared_ollama.api.routes.models import router as models_router
from shared_ollama.api.routes.system import router as system_router
from shared_ollama.api.routes.vlm import router as vlm_router

__all__ = [
    "agents_router",
    "batch_router",
    "chat_router",
    "embeddings_router",
    "generation_router",
    "models_router",
    "system_router",
    "vlm_router",
]
