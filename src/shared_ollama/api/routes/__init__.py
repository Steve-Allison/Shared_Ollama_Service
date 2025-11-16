"""API routes for the Shared Ollama Service.

Modular route definitions organized by functionality.
"""

from shared_ollama.api.routes.batch import router as batch_router
from shared_ollama.api.routes.chat import router as chat_router
from shared_ollama.api.routes.generation import router as generation_router
from shared_ollama.api.routes.system import router as system_router
from shared_ollama.api.routes.vlm import router as vlm_router

__all__ = [
    "system_router",
    "generation_router",
    "chat_router",
    "vlm_router",
    "batch_router",
]
