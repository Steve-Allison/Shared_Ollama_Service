"""Model management routes for Ollama models.

Provides endpoints for managing Ollama models: showing details, creating custom models,
copying models, and listing running models.

Key Features:
    - Model Information: Get detailed model information
    - Model Creation: Create custom models from Modelfiles
    - Model Copying: Duplicate existing models
    - Running Models: List currently loaded models
    - Error Handling: Comprehensive error handling

Endpoints:
    GET /api/v1/models/{name}/show - Get detailed model information
    POST /api/v1/models/create - Create a custom model
    POST /api/v1/models/{name}/copy - Copy a model
    GET /api/v1/models/ps - List running models
"""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from shared_ollama.api.middleware import limiter
from shared_ollama.infrastructure.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models/{name}/show", tags=["Models"])
@limiter.limit("30/minute")
async def show_model(
    name: str = Path(..., description="Model name"),
    request: Request = None,  # noqa: ARG001
) -> dict:
    """Get detailed information about a model.

    Retrieves comprehensive model information including Modelfile, parameters,
    and metadata.

    Args:
        name: Model name to show information for.

    Returns:
        Dictionary with model details including:
            - modelfile: Modelfile content
            - parameters: Model parameters
            - template: Template used
            - details: Model details (size, format, etc.)

    Raises:
        HTTPException: If model not found or request fails.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama.base_url}/api/show",
                json={"name": name},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{name}' not found",
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get model information",
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Request error showing model %s: %s", name, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service unavailable",
            ) from exc


@router.post("/models/create", tags=["Models"])
@limiter.limit("10/minute")
async def create_model(
    request: Request,
) -> dict:
    """Create a custom model from a Modelfile.

    Creates a new model using a Modelfile definition. The request body should
    contain 'name' and 'modelfile' fields.

    Args:
        request: FastAPI Request object with JSON body containing:
            - name: Model name
            - modelfile: Modelfile content (string)
            - stream: Optional, whether to stream progress

    Returns:
        Dictionary with creation status and progress.

    Raises:
        HTTPException: If creation fails or invalid request.
    """
    body = await request.json()
    name = body.get("name")
    modelfile = body.get("modelfile")
    stream = body.get("stream", False)

    if not name or not modelfile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: 'name' and 'modelfile'",
        )

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama.base_url}/api/create",
                json={"name": name, "modelfile": modelfile, "stream": stream},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create model: {exc.response.text}",
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Request error creating model: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service unavailable",
            ) from exc


@router.post("/models/{name}/copy", tags=["Models"])
@limiter.limit("10/minute")
async def copy_model(
    name: str = Path(..., description="Source model name"),
    request: Request = None,  # noqa: ARG001
) -> dict:
    """Copy a model to a new name.

    Creates a copy of an existing model with a new name.

    Args:
        name: Source model name to copy.
        request: FastAPI Request object with JSON body containing:
            - destination: New model name

    Returns:
        Dictionary with copy status.

    Raises:
        HTTPException: If copy fails or model not found.
    """
    body = await request.json()
    destination = body.get("destination")

    if not destination:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required field: 'destination'",
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama.base_url}/api/copy",
                json={"source": name, "destination": destination},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{name}' not found",
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to copy model",
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Request error copying model %s: %s", name, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service unavailable",
            ) from exc


@router.get("/models/ps", tags=["Models"])
@limiter.limit("30/minute")
async def list_running_models(
    request: Request = None,  # noqa: ARG001
) -> dict:
    """List currently running/loaded models.

    Returns information about models currently loaded in memory, including
    their memory usage and load times.

    Returns:
        Dictionary with 'models' list containing:
            - name: Model name
            - size: Model size in bytes
            - size_vram: VRAM usage in bytes
            - digest: Model digest
            - details: Model details

    Raises:
        HTTPException: If request fails.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{settings.ollama.base_url}/api/ps")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            logger.exception("Request error listing running models: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service unavailable",
            ) from exc

