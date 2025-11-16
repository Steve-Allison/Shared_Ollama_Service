"""API routes for the Shared Ollama Service.

Modular route definitions organized by functionality.
"""

from shared_ollama.api.routes.system import router as system_router

__all__ = ["system_router"]
