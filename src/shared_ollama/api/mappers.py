"""Mappers between API models and domain entities.

This module provides conversion functions between FastAPI request/response
models (outer layer) and domain entities (inner layer), following Clean
Architecture boundaries.

All mapping logic is isolated here to maintain separation of concerns.
"""

from __future__ import annotations

from shared_ollama.api.models import (
    ChatMessage as APIChatMessage,
    ChatRequest as APIChatRequest,
    GenerateRequest as APIGenerateRequest,
    ImageContentPart,
    ModelInfo as APIModelInfo,
    TextContentPart,
)
from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    GenerationRequest,
    ImageContent,
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

    Handles both text-only and multimodal (text + images) content.

    Args:
        api_req: API request model.

    Returns:
        Domain ChatRequest entity.

    Raises:
        ValueError: If request validation fails (handled by domain entity).
    """
    domain_messages: list[ChatMessage] = []
    for api_msg in api_req.messages:
        if isinstance(api_msg.content, str):
            # Text-only message (backward compatible)
            domain_messages.append(ChatMessage(role=api_msg.role, content=api_msg.content))
        else:
            # Multimodal message - convert content parts
            content_parts: list[tuple[str, str | ImageContent]] = []
            for part in api_msg.content:
                if isinstance(part, TextContentPart):
                    content_parts.append(("text", part.text))
                elif isinstance(part, ImageContentPart):
                    image_url = part.image_url["url"]
                    content_parts.append(("image_url", ImageContent(url=image_url)))
            domain_messages.append(ChatMessage(role=api_msg.role, content=tuple(content_parts)))

    messages = tuple(domain_messages)
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
