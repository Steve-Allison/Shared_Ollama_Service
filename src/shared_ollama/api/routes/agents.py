"""Agent routes for Ollama 0.13.5 agent system.

Provides endpoints for managing and using Ollama agents. Agents are specialized
AI assistants with predefined roles and behaviors.

Key Features:
    - Agent Management: List available agents, create custom agents
    - Agent Execution: Run agents with prompts and tools
    - Agent Templates: Use predefined agent templates
    - Error Handling: Comprehensive error handling

Endpoints:
    GET /api/v1/agents - List available agents
    POST /api/v1/agents/{name}/run - Run an agent
    POST /api/v1/agents/create - Create a custom agent
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


@router.get("/agents", tags=["Agents"])
@limiter.limit("30/minute")
async def list_agents(
    request: Request = None,  # noqa: ARG001
) -> dict:
    """List available agents.

    Returns list of available agent templates and custom agents.

    Returns:
        Dictionary with 'agents' list containing agent information.

    Note:
        Agent system is new in Ollama 0.13.5. This endpoint may return
        empty list if agents are not yet configured.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Note: Ollama 0.13.5 agent API endpoint may vary
            # This is a placeholder for the actual agent listing endpoint
            # Check Ollama docs for the correct endpoint when available
            response = await client.get(f"{settings.ollama.base_url}/api/agents")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Agents endpoint may not exist yet
                return {"agents": [], "message": "Agent system not available"}
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list agents",
            ) from exc
        except httpx.RequestError as exc:
            logger.debug("Request error listing agents: %s", exc)
            return {"agents": [], "message": "Agent system not available"}


@router.post("/agents/{name}/run", tags=["Agents"])
@limiter.limit("60/minute")
async def run_agent(
    name: str = Path(..., description="Agent name"),
    request: Request = None,  # noqa: ARG001
) -> dict:
    """Run an agent with a prompt.

    Executes an agent with the provided prompt and returns the agent's response.

    Args:
        name: Agent name to run.
        request: FastAPI Request object with JSON body containing:
            - prompt: Prompt for the agent
            - tools: Optional list of tools available to agent
            - options: Optional generation options

    Returns:
        Dictionary with agent response and metadata.

    Raises:
        HTTPException: If agent execution fails.
    """
    body = await request.json()
    prompt = body.get("prompt")
    tools = body.get("tools")
    options = body.get("options")

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required field: 'prompt'",
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            payload = {"agent": name, "prompt": prompt}
            if tools:
                payload["tools"] = tools
            if options:
                payload["options"] = options

            # Note: Actual endpoint may differ - check Ollama 0.13.5 docs
            response = await client.post(
                f"{settings.ollama.base_url}/api/agent/run",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to run agent: {exc.response.text}",
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Request error running agent %s: %s", name, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service unavailable",
            ) from exc


@router.post("/agents/create", tags=["Agents"])
@limiter.limit("10/minute")
async def create_agent(
    request: Request,
) -> dict:
    """Create a custom agent.

    Creates a new agent with custom role and behavior configuration.

    Args:
        request: FastAPI Request object with JSON body containing:
            - name: Agent name
            - role: Agent role/description
            - model: Base model to use
            - tools: Optional list of tools
            - config: Optional agent configuration

    Returns:
        Dictionary with creation status.

    Raises:
        HTTPException: If creation fails.
    """
    body = await request.json()
    name = body.get("name")
    role = body.get("role")
    model = body.get("model")

    if not name or not role or not model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields: 'name', 'role', 'model'",
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Note: Actual endpoint may differ - check Ollama 0.13.5 docs
            payload = {"name": name, "role": role, "model": model}
            if "tools" in body:
                payload["tools"] = body["tools"]
            if "config" in body:
                payload["config"] = body["config"]

            response = await client.post(
                f"{settings.ollama.base_url}/api/agent/create",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create agent: {exc.response.text}",
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Request error creating agent: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service unavailable",
            ) from exc

