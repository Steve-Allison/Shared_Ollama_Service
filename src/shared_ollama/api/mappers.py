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
    ResponseFormat,
    Tool as APITool,
    ToolCall as APIToolCall,
    ToolCallFunction as APIToolCallFunction,
    ToolFunction as APIToolFunction,
)
from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    GenerationRequest,
    ModelInfo,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
)
from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage


# ============================================================================
# Tool Calling Mappers
# ============================================================================


def api_to_domain_tool(api_tool: APITool) -> Tool:
    """Convert API Tool to domain Tool.

    Args:
        api_tool: API Tool model.

    Returns:
        Domain Tool entity.
    """
    function = ToolFunction(
        name=api_tool.function.name,
        description=api_tool.function.description,
        parameters=api_tool.function.parameters,
    )
    return Tool(function=function, type=api_tool.type)


def api_to_domain_tool_call(api_tool_call: APIToolCall) -> ToolCall:
    """Convert API ToolCall to domain ToolCall.

    Args:
        api_tool_call: API ToolCall model.

    Returns:
        Domain ToolCall entity.
    """
    function = ToolCallFunction(
        name=api_tool_call.function.name,
        arguments=api_tool_call.function.arguments,
    )
    return ToolCall(id=api_tool_call.id, function=function, type=api_tool_call.type)


def domain_to_api_tool_call(domain_tool_call: ToolCall) -> APIToolCall:
    """Convert domain ToolCall to API ToolCall.

    Args:
        domain_tool_call: Domain ToolCall entity.

    Returns:
        API ToolCall model.
    """
    function = APIToolCallFunction(
        name=domain_tool_call.function.name,
        arguments=domain_tool_call.function.arguments,
    )
    return APIToolCall(id=domain_tool_call.id, function=function, type=domain_tool_call.type)


# ============================================================================
# Response Format Helpers
# ============================================================================


def _resolve_response_format(
    direct_format: str | dict[str, Any] | None,
    response_format: ResponseFormat | None,
) -> str | dict[str, Any] | None:
    """Resolve native format + OpenAI response_format into Ollama format payload.

    Uses match/case for cleaner pattern matching (Python 3.13+).

    Args:
        direct_format: Native Ollama format field (deprecated).
        response_format: OpenAI-compatible response_format field.

    Returns:
        Resolved format value for Ollama backend.
    """
    # Guard clause: if no response_format, use direct format
    if response_format is None:
        return direct_format

    # Use match/case for response format type resolution
    match response_format.type:
        case "json_object":
            return "json"

        case "json_schema":
            schema_payload: dict[str, Any] | None = response_format.json_schema
            if not schema_payload:
                raise ValueError("response_format.json_schema is required when type='json_schema'")

            # OpenAI wraps schema as {"name": "...", "schema": {...}}.
            match schema_payload:
                case {"schema": dict() as nested_schema}:
                    return nested_schema
                case _:
                    return schema_payload

        case "text" | _:
            # Fall back to native field for text or unknown types
            return None


# ============================================================================
# Generation Request Mappers
# ============================================================================


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
    # Build options only if any option is provided (performance: use generator)
    if any(
        api_req.temperature is not None,
        api_req.top_p is not None,
        api_req.top_k is not None,
        api_req.max_tokens is not None,
        api_req.seed is not None,
        api_req.stop is not None,
    ):
        options = GenerationOptions(
            temperature=api_req.temperature or 0.2,
            top_p=api_req.top_p or 0.9,
            top_k=api_req.top_k or 40,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    # Convert tools if present
    tools = None
    if api_req.tools:
        tools = tuple(api_to_domain_tool(t) for t in api_req.tools)

    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    return GenerationRequest(
        prompt=prompt,
        model=model,
        system=system,
        options=options,
        format=resolved_format,
        tools=tools,
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
    # Convert messages with tool calling support
    domain_messages: list[ChatMessage] = []
    for api_msg in api_req.messages:
        # Convert tool calls if present
        tool_calls = None
        if api_msg.tool_calls:
            tool_calls = tuple(api_to_domain_tool_call(tc) for tc in api_msg.tool_calls)

        domain_messages.append(
            ChatMessage(
                role=api_msg.role,
                content=api_msg.content,
                tool_calls=tool_calls,
                tool_call_id=api_msg.tool_call_id,
            )
        )

    messages = tuple(domain_messages)
    model = ModelName(value=api_req.model) if api_req.model else None

    options: GenerationOptions | None = None
    # Build options only if any option is provided (performance: use generator)
    if any(
        api_req.temperature is not None,
        api_req.top_p is not None,
        api_req.top_k is not None,
        api_req.max_tokens is not None,
        api_req.seed is not None,
        api_req.stop is not None,
    ):
        options = GenerationOptions(
            temperature=api_req.temperature or 0.2,
            top_p=api_req.top_p or 0.9,
            top_k=api_req.top_k or 40,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    # Convert tools if present
    tools = None
    if api_req.tools:
        tools = tuple(api_to_domain_tool(t) for t in api_req.tools)

    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    return ChatRequest(
        messages=messages,
        model=model,
        options=options,
        format=resolved_format,
        tools=tools,
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

    # Convert API messages to domain messages with tool calling support
    messages: list[VLMMessage] = []
    for msg in api_req.messages:
        # Convert tool calls if present
        tool_calls = None
        if msg.tool_calls:
            tool_calls = tuple(api_to_domain_tool_call(tc) for tc in msg.tool_calls)

        domain_msg = VLMMessage(
            role=msg.role,
            content=msg.content,
            tool_calls=tool_calls,
            tool_call_id=msg.tool_call_id,
        )
        messages.append(domain_msg)

    # Convert model name
    model = ModelName(api_req.model) if api_req.model else None

    # Convert options
    options = None
    # Convert options (performance: use generator instead of list)
    if any(
        api_req.temperature is not None,
        api_req.top_p is not None,
        api_req.top_k is not None,
        api_req.max_tokens is not None,
        api_req.seed is not None,
        api_req.stop is not None,
    ):
        options = GenerationOptions(
            temperature=api_req.temperature,
            top_p=api_req.top_p,
            top_k=api_req.top_k,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    # Convert tools if present
    tools = None
    if api_req.tools:
        tools = tuple(api_to_domain_tool(t) for t in api_req.tools)

    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    return VLMRequest(
        messages=tuple(messages),
        images=tuple(api_req.images),  # Native Ollama format: separate images
        model=model,
        options=options,
        image_compression=api_req.image_compression,
        max_dimension=api_req.max_dimension,
        format=resolved_format,
        tools=tools,
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
        ImageContentPart,
        TextContentPart,
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
    # Convert options (performance: use generator instead of list)
    if any(
        api_req.temperature is not None,
        api_req.top_p is not None,
        api_req.top_k is not None,
        api_req.max_tokens is not None,
        api_req.seed is not None,
        api_req.stop is not None,
    ):
        options = GenerationOptions(
            temperature=api_req.temperature,
            top_p=api_req.top_p,
            top_k=api_req.top_k,
            max_tokens=api_req.max_tokens,
            seed=api_req.seed,
            stop=api_req.stop,
        )

    # Convert tools if present
    tools = None
    if api_req.tools:
        tools = tuple(api_to_domain_tool(t) for t in api_req.tools)

    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    # Return native Ollama format (text-only messages + separate images)
    return VLMRequest(
        messages=tuple(messages),
        images=tuple(images),  # Images extracted from multimodal content
        model=model,
        options=options,
        image_compression=api_req.image_compression,
        max_dimension=api_req.max_dimension,
        format=resolved_format,
        tools=tools,
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
