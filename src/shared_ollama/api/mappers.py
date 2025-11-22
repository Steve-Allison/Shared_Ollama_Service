"""Mappers between API models and domain entities.

This module provides conversion functions between FastAPI request/response
models (outer layer) and domain entities (inner layer), following Clean
Architecture boundaries.

Design Principles:
    - Unidirectional: API -> Domain (requests) and Domain -> API (responses)
    - Isolated: All mapping logic centralized in this module
    - Type-safe: Explicit conversions with validation
    - Format-aware: Handles OpenAI vs native Ollama format differences

Key Mappers:
    - api_to_domain_*: Convert API models to domain entities
    - domain_to_api_*: Convert domain entities to API models
    - Tool calling: Bidirectional conversion for POML compatibility
    - VLM requests: Handles both native Ollama and OpenAI formats

Note:
    Mappers handle format differences between API layer (FastAPI/Pydantic)
    and domain layer (pure Python dataclasses). They also handle OpenAI
    format conversion for compatibility with OpenAI-compatible clients.
"""

from __future__ import annotations

from typing import Any

from shared_ollama.api.models import (
    ChatRequest as APIChatRequest,
    GenerateRequest as APIGenerateRequest,
    ImageContentPart,
    ModelInfo as APIModelInfo,
    ResponseFormat,
    TextContentPart,
    Tool as APITool,
    ToolCall as APIToolCall,
    ToolCallFunction as APIToolCallFunction,
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
    VLMMessage,
)
from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage

# ============================================================================
# Tool Calling Mappers
# ============================================================================


def api_to_domain_tool(api_tool: APITool) -> Tool:
    """Convert API Tool to domain Tool.

    Transforms a Pydantic API Tool model to a domain Tool entity,
    preserving function definition and type information.

    Args:
        api_tool: API Tool model from FastAPI request. Contains function
            definition with name, description, and parameters schema.

    Returns:
        Domain Tool entity with ToolFunction and type. Domain entity
        validates itself during construction.

    Note:
        Tool type is preserved (currently always "function") for future
        extensibility. Function definition is converted to ToolFunction
        value object.
    """
    function = ToolFunction(
        name=api_tool.function.name,
        description=api_tool.function.description,
        parameters=api_tool.function.parameters,
    )
    return Tool(function=function, type=api_tool.type)


def api_to_domain_tool_call(api_tool_call: APIToolCall) -> ToolCall:
    """Convert API ToolCall to domain ToolCall.

    Transforms a Pydantic API ToolCall model to a domain ToolCall entity,
    preserving call ID, function name, and arguments.

    Args:
        api_tool_call: API ToolCall model from FastAPI request. Contains
            tool call ID, function name, and JSON-serialized arguments.

    Returns:
        Domain ToolCall entity with ToolCallFunction. Domain entity
        validates itself during construction.

    Note:
        Arguments remain as JSON string in domain entity (matching OpenAI
        and POML formats). Parse with json.loads() when executing the function.
    """
    function = ToolCallFunction(
        name=api_tool_call.function.name,
        arguments=api_tool_call.function.arguments,
    )
    return ToolCall(id=api_tool_call.id, function=function, type=api_tool_call.type)


def domain_to_api_tool_call(domain_tool_call: ToolCall) -> APIToolCall:
    """Convert domain ToolCall to API ToolCall.

    Transforms a domain ToolCall entity to a Pydantic API ToolCall model
    for API responses.

    Args:
        domain_tool_call: Domain ToolCall entity from model generation.
            Contains tool call ID, function name, and JSON-serialized arguments.

    Returns:
        API ToolCall model suitable for FastAPI response serialization.

    Note:
        This mapper is used when models generate tool calls that need to
        be returned in API responses (e.g., assistant messages with tool_calls).
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

    Handles format resolution from both native Ollama format field and
    OpenAI-compatible response_format field. OpenAI format takes precedence
    when both are provided.

    Format Resolution:
        - If response_format is None: Use direct_format
        - If response_format.type is "json_object": Return "json"
        - If response_format.type is "json_schema": Extract and return schema
        - If response_format.type is "text": Use direct_format
        - Unknown types: Use direct_format

    Args:
        direct_format: Native Ollama format field. Can be "json", dict schema,
            or None. Deprecated in favor of response_format.
        response_format: OpenAI-compatible response_format object. Contains
            type and optional json_schema. Takes precedence over direct_format.

    Returns:
        Resolved format value for Ollama backend. Can be:
            - "json": For JSON mode
            - dict: JSON schema for structured output
            - None: Default text output (no constraints)

    Raises:
        ValueError: If response_format.type is "json_schema" but json_schema
            is None or missing.

    Note:
        OpenAI wraps JSON schemas as {"name": "...", "schema": {...}}. This
        function extracts the inner schema if present, otherwise uses the
        payload directly.
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
# Generation Options Helper
# ============================================================================


def _build_generation_options(
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int | None,
    seed: int | None,
    stop: list[str] | None,
    *,
    use_defaults: bool = True,
) -> GenerationOptions | None:
    """Build GenerationOptions from API fields with optional defaults.

    Centralizes options building logic for all request mappers. Handles both
    native Ollama format (with defaults) and OpenAI format (without defaults).

    Args:
        temperature: Sampling temperature (0.0-2.0).
        top_p: Top-p (nucleus) sampling parameter (0.0-1.0).
        top_k: Top-k sampling parameter.
        max_tokens: Maximum tokens to generate.
        seed: Random seed for reproducibility.
        stop: Stop sequences to halt generation.
        use_defaults: Whether to apply defaults for temperature/top_p/top_k.
            True for native Ollama format, False for OpenAI format.

    Returns:
        GenerationOptions entity if any option is provided, None otherwise.

    Note:
        When use_defaults=True: temperature=0.2, top_p=0.9, top_k=40
        When use_defaults=False: Only explicitly provided values are set
    """
    # Early return if no options provided
    if not any((temperature, top_p, top_k, max_tokens, seed, stop)):
        return None

    if use_defaults:
        return GenerationOptions(
            temperature=temperature or 0.2,
            top_p=top_p or 0.9,
            top_k=top_k or 40,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
        )

    # Build options dict with only provided values (OpenAI format)
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if seed is not None:
        kwargs["seed"] = seed
    if stop is not None:
        kwargs["stop"] = stop

    return GenerationOptions(**kwargs) if kwargs else None


# ============================================================================
# Generation Request Mappers
# ============================================================================


def api_to_domain_generation_request(api_req: APIGenerateRequest) -> GenerationRequest:
    """Convert API GenerateRequest to domain GenerationRequest.

    Transforms a Pydantic API request model to a domain entity, converting
    primitive types to value objects and building domain structures.

    Args:
        api_req: API GenerateRequest model from FastAPI endpoint. Contains
            prompt, optional model, system, options, format, and tools.

    Returns:
        Domain GenerationRequest entity with validated value objects.
        Domain entity validates itself during construction.

    Raises:
        ValueError: If request validation fails. Validation is performed by:
            - Prompt value object (empty/whitespace check, length limit)
            - ModelName value object (empty check)
            - GenerationOptions (parameter range validation)
            - GenerationRequest (prompt validation)

    Note:
        Options are only created if at least one option is provided (performance
        optimization). Tools are converted to domain Tool entities. Format is
        resolved from both native format and OpenAI response_format fields.
    """
    prompt = Prompt(value=api_req.prompt)
    model = ModelName(value=api_req.model) if api_req.model else None
    system = SystemMessage(value=api_req.system) if api_req.system else None

    options = _build_generation_options(
        temperature=api_req.temperature,
        top_p=api_req.top_p,
        top_k=api_req.top_k,
        max_tokens=api_req.max_tokens,
        seed=api_req.seed,
        stop=api_req.stop,
    )

    tools = tuple(api_to_domain_tool(t) for t in api_req.tools) if api_req.tools else None
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

    Transforms a Pydantic API chat request model to a domain entity, converting
    messages with tool calling support and building domain structures.

    Args:
        api_req: API ChatRequest model from FastAPI endpoint. Contains messages,
            optional model, options, format, and tools. Messages support tool
            calling (tool_calls, tool_call_id).

    Returns:
        Domain ChatRequest entity with validated ChatMessage entities.
        Domain entity validates itself during construction.

    Raises:
        ValueError: If request validation fails. Validation is performed by:
            - ChatMessage entities (role validation, content/tool_calls requirement)
            - ChatRequest (non-empty messages, total length limit)

    Note:
        Messages are converted with full tool calling support. Tool calls from
        assistant messages and tool responses are preserved. Options and tools
        are handled the same way as generation requests.
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

    options = _build_generation_options(
        temperature=api_req.temperature,
        top_p=api_req.top_p,
        top_k=api_req.top_k,
        max_tokens=api_req.max_tokens,
        seed=api_req.seed,
        stop=api_req.stop,
    )

    tools = tuple(api_to_domain_tool(t) for t in api_req.tools) if api_req.tools else None
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
        ModelName,
        VLMMessage,
        VLMRequest,
    )

    # Convert API messages to domain messages with tool calling support
    messages: list[VLMMessage] = []
    # Find the last user message to attach images to (iterate backwards efficiently)
    last_user_msg_idx = -1
    for i in range(len(api_req.messages) - 1, -1, -1):
        if api_req.messages[i].role == "user":
            last_user_msg_idx = i
            break

    for i, msg in enumerate(api_req.messages):
        tool_calls = (
            tuple(api_to_domain_tool_call(tc) for tc in msg.tool_calls) if msg.tool_calls else None
        )
        images = tuple(api_req.images) if i == last_user_msg_idx else None

        messages.append(
            VLMMessage(
                role=msg.role,
                content=msg.content,
                images=images,
                tool_calls=tool_calls,
                tool_call_id=msg.tool_call_id,
            )
        )

    model = ModelName(value=api_req.model) if api_req.model else None

    options = _build_generation_options(
        temperature=api_req.temperature,
        top_p=api_req.top_p,
        top_k=api_req.top_k,
        max_tokens=api_req.max_tokens,
        seed=api_req.seed,
        stop=api_req.stop,
        use_defaults=False,
    )

    tools = None
    if api_req.tools:
        tools = tuple(api_to_domain_tool(t) for t in api_req.tools)

    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    return VLMRequest(
        messages=tuple(messages),
        model=model,
        options=options,
        image_compression=api_req.image_compression,
        max_dimension=api_req.max_dimension,
        format=resolved_format,
        tools=tools,
    )


def _extract_vlm_messages(api_req: Any) -> tuple[VLMMessage, ...]:
    """Extract VLM messages from OpenAI-format request.

    Converts OpenAI-compatible multimodal messages to native VLMMessage format,
    handling role normalization and image extraction.

    Args:
        api_req: OpenAI-compatible VLM request with multimodal messages.

    Returns:
        Tuple of VLMMessage entities ready for processing.
    """
    messages: list[VLMMessage] = []

    for msg in api_req.messages:
        text_parts: list[str] = []
        images: list[str] = []
        content = msg.content
        if isinstance(content, str):
            text_parts.append(content)
        else:
            for part in content or []:
                if isinstance(part, TextContentPart):
                    text_parts.append(part.text)
                elif isinstance(part, ImageContentPart):
                    images.append(part.image_url.url)

        # Join text parts, or use default prompt for image-only messages
        text_content = (
            " ".join(text_parts) if text_parts else ("Describe this image." if images else None)
        )

        # Normalize role: "developer" -> "system" for Ollama compatibility
        normalized_role = _normalize_openai_role(msg.role)

        messages.append(
            VLMMessage(
                role=normalized_role,  # type: ignore[arg-type] - validated by OpenAI API
                content=text_content,
                images=tuple(images) if images else None,
            )
        )

    return tuple(messages)


def _normalize_openai_role(role: str) -> str:
    """Normalize OpenAI role to Ollama-compatible role.

    OpenAI uses "developer" role for system instructions in newer models.
    Ollama uses "system" role for the same purpose.

    Args:
        role: OpenAI message role.

    Returns:
        Ollama-compatible role string.
    """
    if role == "developer":
        return "system"
    return role


def api_to_domain_chat_request_openai(api_req: Any) -> ChatRequest:
    """Convert OpenAI-compatible API ChatRequest to native Ollama domain ChatRequest.

    Transforms an OpenAI-compatible chat request (ChatRequestOpenAI) to a domain
    entity. Handles text-only messages from OpenAI format (string content) and
    converts them to native Ollama ChatMessage format.

    Args:
        api_req: OpenAI-compatible ChatRequestOpenAI model. Contains messages
            with string content (text-only), optional model, options, format, and tools.

    Returns:
        Domain ChatRequest entity with validated ChatMessage entities.
        Domain entity validates itself during construction.

    Raises:
        ValueError: If request validation fails. Validation is performed by:
            - ChatMessage entities (role validation, content requirement)
            - ChatRequest (non-empty messages, total length limit)

    Note:
        OpenAI format uses string content for text-only messages, which is
        directly converted to native ChatMessage format. The "developer" role
        is converted to "system" for Ollama compatibility. Tool calls and
        other features are preserved.
    """
    from shared_ollama.domain.entities import (
        ChatMessage,
        ChatRequest,
        ModelName,
    )

    # Convert messages from OpenAI format (string content) to native format
    domain_messages: list[ChatMessage] = []
    for api_msg in api_req.messages:
        # OpenAI format: content is string for text-only messages
        content: str | None = None
        if isinstance(api_msg.content, str):
            content = api_msg.content
        elif isinstance(api_msg.content, list):
            text_parts = [
                part.text for part in api_msg.content if hasattr(part, "text") and part.text
            ]
            content = " ".join(text_parts) if text_parts else None

        tool_calls = (
            tuple(api_to_domain_tool_call(tc) for tc in api_msg.tool_calls)
            if api_msg.tool_calls
            else None
        )
        normalized_role = _normalize_openai_role(api_msg.role)

        domain_messages.append(
            ChatMessage(
                role=normalized_role,  # type: ignore[arg-type]
                content=content,
                tool_calls=tool_calls,
                tool_call_id=api_msg.tool_call_id,
            )
        )

    messages = tuple(domain_messages)
    model = ModelName(value=api_req.model) if api_req.model else None

    # Use effective_max_tokens for OpenAI compatibility (prefers max_completion_tokens)
    options = _build_generation_options(
        temperature=api_req.temperature,
        top_p=api_req.top_p,
        top_k=api_req.top_k,
        max_tokens=api_req.effective_max_tokens,
        seed=api_req.seed,
        stop=api_req.stop,
    )

    tools = tuple(api_to_domain_tool(t) for t in api_req.tools) if api_req.tools else None

    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    return ChatRequest(
        messages=messages,
        model=model,
        options=options,
        format=resolved_format,
        tools=tools,
    )


def api_to_domain_vlm_request_openai(api_req: Any) -> Any:
    """Convert OpenAI-compatible API VLMRequest to native Ollama domain VLMRequest."""
    from shared_ollama.domain.entities import (
        ModelName,
        VLMRequest,
    )

    messages = _extract_vlm_messages(api_req)
    model = ModelName(value=api_req.model) if api_req.model else None

    # Use effective_max_tokens for OpenAI compatibility (prefers max_completion_tokens)
    options = _build_generation_options(
        temperature=api_req.temperature,
        top_p=api_req.top_p,
        top_k=api_req.top_k,
        max_tokens=api_req.effective_max_tokens,
        seed=api_req.seed,
        stop=api_req.stop,
        use_defaults=False,
    )

    tools = tuple(api_to_domain_tool(t) for t in api_req.tools) if api_req.tools else None
    resolved_format = _resolve_response_format(api_req.format, api_req.response_format)

    return VLMRequest(
        messages=messages,
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
