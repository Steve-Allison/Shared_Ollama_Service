"""Mappers between API models and domain entities.

This module provides conversion functions between FastAPI request/response
models (outer layer) and domain entities (inner layer), following Clean
Architecture boundaries.

All mapping logic is isolated here to maintain separation of concerns.
"""

from __future__ import annotations

from shared_ollama.api.models import (
    ChatRequest as APIChatRequest,
    GenerateRequest as APIGenerateRequest,
    ModelInfo as APIModelInfo,
)
from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    GenerationRequest,
    ModelInfo,
)
from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage


def api_to_domain_generation_request(api_req: APIGenerateRequest) -> GenerationRequest:
    """Convert API GenerateRequest to domain GenerationRequest.

    Args:
        api_req: API request model.

    Returns:
        Domain GenerationRequest entity.

    Raises:
        ValueError: If request validation fails (handled by domain entity).
    """
    prompt = Prompt(value=api_req.prompt)
    model = ModelName(value=api_req.model) if api_req.model else None
    system = SystemMessage(value=api_req.system) if api_req.system else None

    options: GenerationOptions | None = None
    if any([
        api_req.temperature is not None,
        api_req.top_p is not None,
        api_req.top_k is not None,
        api_req.max_tokens is not None,
        api_req.seed is not None,
        api_req.stop is not None,
    ]):
        options = GenerationOptions(
            temperature=api_req.temperature or 0.2,
            top_p=api_req.top_p or 0.9,
            top_k=api_req.top_k or 40,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    return GenerationRequest(
        prompt=prompt,
        model=model,
        system=system,
        options=options,
        format=api_req.format,
    )


def api_to_domain_chat_request(api_req: APIChatRequest) -> ChatRequest:
    """Convert API ChatRequest to domain ChatRequest.

    Args:
        api_req: API request model.

    Returns:
        Domain ChatRequest entity.

    Raises:
        ValueError: If request validation fails (handled by domain entity).
    """
    messages = tuple(ChatMessage(role=msg.role, content=msg.content) for msg in api_req.messages)
    model = ModelName(value=api_req.model) if api_req.model else None

    options: GenerationOptions | None = None
    if any([
        api_req.temperature is not None,
        api_req.top_p is not None,
        api_req.top_k is not None,
        api_req.max_tokens is not None,
        api_req.seed is not None,
        api_req.stop is not None,
    ]):
        options = GenerationOptions(
            temperature=api_req.temperature or 0.2,
            top_p=api_req.top_p or 0.9,
            top_k=api_req.top_k or 40,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    return ChatRequest(
        messages=messages,
        model=model,
        options=options,
    )


def domain_to_api_model_info(domain_model: ModelInfo) -> APIModelInfo:
    """Convert domain ModelInfo to API ModelInfo.

    Args:
        domain_model: Domain ModelInfo entity.

    Returns:
        API ModelInfo model.
    """
    return APIModelInfo(
        name=domain_model.name,
        size=domain_model.size,
        modified_at=domain_model.modified_at,
    )
