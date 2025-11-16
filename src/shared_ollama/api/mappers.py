"""Mappers between API models and domain entities.

This module provides conversion functions between FastAPI request/response
models (outer layer) and domain entities (inner layer), following Clean
Architecture boundaries.

All mapping logic is isolated here to maintain separation of concerns.
"""

from __future__ import annotations

from typing import Any

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
    """Convert API ChatRequest to domain ChatRequest (text-only, native Ollama).

    Args:
        api_req: API request model (text-only).

    Returns:
        Domain ChatRequest entity.

    Raises:
        ValueError: If request validation fails (handled by domain entity).
    """
    # Convert text-only messages (native Ollama format)
    domain_messages: list[ChatMessage] = []
    for api_msg in api_req.messages:
        domain_messages.append(ChatMessage(role=api_msg.role, content=api_msg.content))

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


def api_to_domain_vlm_request(api_req: Any) -> Any:  # VLMRequest from models
    """Convert API VLMRequest to domain VLMRequest (native Ollama format).

    Args:
        api_req: API VLMRequest model (native Ollama format with separate images).

    Returns:
        Domain VLMRequest entity.

    Raises:
        ValueError: If conversion fails.
    """
    from shared_ollama.domain.entities import (
        GenerationOptions,
        ModelName,
        VLMMessage,
        VLMRequest,
    )

    # Convert API messages to domain messages (text-only, native Ollama format)
    messages: list[VLMMessage] = []
    for msg in api_req.messages:
        domain_msg = VLMMessage(role=msg.role, content=msg.content)
        messages.append(domain_msg)

    # Convert model name
    model = ModelName(api_req.model) if api_req.model else None

    # Convert options
    options = None
    if any(
        [
            api_req.temperature,
            api_req.top_p,
            api_req.top_k,
            api_req.max_tokens,
            api_req.seed,
            api_req.stop,
        ]
    ):
        options = GenerationOptions(
            temperature=api_req.temperature,
            top_p=api_req.top_p,
            top_k=api_req.top_k,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    return VLMRequest(
        messages=tuple(messages),
        images=tuple(api_req.images),  # Native Ollama format: separate images
        model=model,
        options=options,
        image_compression=api_req.image_compression,
        max_dimension=api_req.max_dimension,
    )


def api_to_domain_vlm_request_openai(api_req: Any) -> Any:
    """Convert OpenAI-compatible API VLMRequest to native Ollama domain VLMRequest.

    Takes OpenAI-compatible format (multimodal messages with embedded images)
    and converts to native Ollama format (text-only messages + separate images).

    Args:
        api_req: API VLMRequestOpenAI with multimodal messages.

    Returns:
        Domain VLMRequest in native Ollama format (text-only messages + separate images).
    """
    from shared_ollama.api.models import (
        ChatMessageOpenAI,
        ImageContentPart,
        TextContentPart,
        VLMRequestOpenAI,
    )
    from shared_ollama.domain.entities import (
        GenerationOptions,
        ModelName,
        VLMMessage,
        VLMRequest,
    )

    # Extract images and build text-only messages
    messages: list[VLMMessage] = []
    images: list[str] = []

    for msg in api_req.messages:
        # Build text content from message
        text_parts: list[str] = []

        if isinstance(msg.content, str):
            # Simple string content
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            # Multimodal content - extract text and images
            for part in msg.content:
                if isinstance(part, TextContentPart):
                    text_parts.append(part.text)
                elif isinstance(part, ImageContentPart):
                    # Extract image URL and add to images list
                    images.append(part.image_url.url)

        # Create text-only message (combine all text parts)
        if text_parts:
            combined_text = " ".join(text_parts)
            messages.append(VLMMessage(role=msg.role, content=combined_text))
        elif not text_parts and images:
            # Message has only images, add a default prompt
            messages.append(VLMMessage(role=msg.role, content="What's in this image?"))

    # Convert model name
    model = ModelName(api_req.model) if api_req.model else None

    # Convert options
    options = None
    if any(
        [
            api_req.temperature,
            api_req.top_p,
            api_req.top_k,
            api_req.max_tokens,
            api_req.seed,
            api_req.stop,
        ]
    ):
        options = GenerationOptions(
            temperature=api_req.temperature,
            top_p=api_req.top_p,
            top_k=api_req.top_k,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    # Return native Ollama format (text-only messages + separate images)
    return VLMRequest(
        messages=tuple(messages),
        images=tuple(images),  # Images extracted from multimodal content
        model=model,
        options=options,
        image_compression=api_req.image_compression,
        max_dimension=api_req.max_dimension,
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
