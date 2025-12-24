"""Embeddings routes for vector embedding generation.

Provides the /embeddings endpoint for generating vector embeddings from text.
Embeddings are useful for semantic search, RAG systems, and similarity matching.

Key Features:
    - Vector Embeddings: Generate embeddings for semantic search
    - Model Selection: Choose appropriate embedding model
    - Error Handling: Comprehensive error handling with structured logging
    - Rate Limiting: Integrated with slowapi rate limiting

Endpoint:
    POST /api/v1/embeddings
        - Request: EmbeddingsRequest (prompt, model)
        - Response: EmbeddingsResponse (embedding vector, model, prompt)
        - Rate Limited: Yes (via slowapi middleware)
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from shared_ollama.api.dependencies import (
    get_request_context,
    parse_request_json,
    validate_model_allowed,
)
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import EmbeddingsRequest, EmbeddingsResponse
from shared_ollama.api.response_builders import json_response
from shared_ollama.client.sync import OllamaConfig, SharedOllamaClient
from shared_ollama.core.utils import get_default_text_model

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/embeddings", response_model=EmbeddingsResponse, tags=["Embeddings"])
@limiter.limit("60/minute")
async def embeddings(
    request: Request,
) -> EmbeddingsResponse:
    """Generate embeddings for a text prompt.

    Creates vector embeddings from text, useful for semantic search, RAG systems,
    and similarity matching.

    Args:
        request: FastAPI Request object (injected). Body must contain
            EmbeddingsRequest JSON with prompt and optional model.

    Returns:
        EmbeddingsResponse with embedding vector, model name, and original prompt.

    Raises:
        HTTPException: If request validation fails or embeddings generation fails.
    """
    # Parse and validate request
    body = await parse_request_json(request)
    embeddings_request = EmbeddingsRequest(**body)

    # Get default model if not specified
    model = embeddings_request.model or get_default_text_model()

    # Validate model is allowed
    validate_model_allowed(model)

    # Get request context for logging
    context = get_request_context(request)

    # Create client and generate embeddings
    config = OllamaConfig(base_url="http://localhost:11434")
    client = SharedOllamaClient(config=config, verify_on_init=False)

    start_time = time.perf_counter()

    try:
        result = client.embeddings(
            prompt=embeddings_request.prompt,
            model=model,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return EmbeddingsResponse(
            embedding=result.get("embedding", []),
            model=result.get("model", model),
            prompt=result.get("prompt", embeddings_request.prompt),
            latency_ms=round(latency_ms, 3),
        )

    except ValueError as exc:
        logger.warning("Invalid embeddings request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to generate embeddings: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings",
        ) from exc

