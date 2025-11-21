"""Domain entities for the Shared Ollama Service.

This module defines pure domain models representing core business concepts
with no framework or infrastructure dependencies. All entities contain business
rules, validation logic, and invariants that enforce domain constraints.

Design Principles:
    - Immutability: All entities are frozen dataclasses (slots=True)
    - Validation: Business rules enforced in __post_init__ methods
    - No I/O: Entities contain no file/network operations
    - Framework-agnostic: No FastAPI, Pydantic, or other framework deps

Key Entities:
    - Model: Enumeration of supported model identifiers
    - ModelInfo: Metadata about available models
    - GenerationRequest/ChatRequest/VLMRequest: Request entities with validation
    - Tool/ToolCall: POML-compatible function calling support
    - ChatMessage/VLMMessage: Message entities with role validation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage

# Domain constants for validation
TEMPERATURE_MAX = 2.0
"""Maximum allowed temperature value for generation (inclusive)."""

PROMPT_MAX_LENGTH = 1_000_000
"""Maximum character length for prompts (inclusive)."""

VALID_ROLES = {"user", "assistant", "system", "tool"}
"""Set of valid message roles for chat/VLM messages."""

MAX_TOTAL_MESSAGE_CHARS = 1_000_000
"""Maximum total character count across all messages in a request."""

MIN_IMAGE_DIMENSION = 256
"""Minimum image dimension in pixels (inclusive)."""

MAX_IMAGE_DIMENSION = 2667
"""Maximum image dimension in pixels (inclusive)."""


class Model(StrEnum):
    """Supported Qwen 3 model identifiers.

    Enumeration of model names used throughout the system. Values are
    string identifiers compatible with Ollama's model naming convention.

    Attributes:
        QWEN3_VL_8B_Q4: Vision-language model, 8B parameters, Q4 quantization.
        QWEN3_14B_Q4: Text-only model, 14B parameters, Q4 quantization.
        QWEN3_VL_32B: Vision-language model, 32B parameters (unquantized).
        QWEN3_30B: Text-only model, 30B parameters (unquantized).
    """

    QWEN3_VL_8B_Q4 = "qwen3-vl:8b-instruct-q4_K_M"
    QWEN3_14B_Q4 = "qwen3:14b-q4_K_M"
    QWEN3_VL_32B = "qwen3-vl:32b"
    QWEN3_30B = "qwen3:30b"


@dataclass(slots=True, frozen=True)
class ModelInfo:
    """Information about an available model.

    Pure domain entity representing model metadata with no I/O or framework
    dependencies. Used to represent models returned from the Ollama service.

    Attributes:
        name: Model name identifier. Must match Ollama model naming convention.
        size: Model size in bytes. None if size information unavailable.
        modified_at: ISO 8601 timestamp of last modification. None if timestamp
            unavailable.

    Note:
        This entity is immutable. All fields are optional except name.
    """

    name: str
    size: int | None = None
    modified_at: str | None = None


# ============================================================================
# Tool Calling Domain Entities (POML Support)
# ============================================================================


@dataclass(slots=True, frozen=True)
class ToolFunction:
    """Function definition for tool calling.

    Represents a callable function that models can invoke. Compatible with
    both POML <tool-definition> and OpenAI function calling formats.

    Attributes:
        name: Function name identifier. Must be non-empty after stripping.
            Used to identify the function when called by the model.
        description: Human-readable function description. Optional but
            recommended for better model understanding.
        parameters: JSON Schema object defining function parameters. None if
            function takes no parameters. Must conform to JSON Schema draft-07.

    Raises:
        ValueError: If name is empty or contains only whitespace.

    Note:
        The parameters dict should follow JSON Schema format:
        {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate tool function.

        Ensures function name is non-empty and non-whitespace.

        Raises:
            ValueError: If name is empty or contains only whitespace.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Function name cannot be empty")


@dataclass(slots=True, frozen=True)
class Tool:
    """Tool definition for function calling.

    Wraps a ToolFunction in a tool container. Currently only supports
    "function" type tools, matching OpenAI and POML conventions.

    Attributes:
        function: Function definition containing name, description, and schema.
        type: Tool type identifier. Always "function" for function calling.
            Reserved for future extensibility (e.g., "code_interpreter").

    Note:
        This entity follows OpenAI's tool format where tools are wrapped
        in a type container. The type field is fixed to "function" for
        compatibility with existing tool calling implementations.
    """

    function: ToolFunction
    type: Literal["function"] = "function"


@dataclass(slots=True, frozen=True)
class ToolCallFunction:
    """Function call details in a tool call.

    Represents the actual invocation of a function by the model, containing
    the function name and serialized arguments.

    Attributes:
        name: Function name being called. Must match a ToolFunction.name from
            the tools list provided to the model.
        arguments: JSON string containing function arguments. Must be valid
            JSON and conform to the function's parameter schema.

    Raises:
        ValueError: If name is empty/whitespace or arguments is empty.

    Note:
        Arguments are stored as a JSON string (not a dict) to match OpenAI
        and POML formats. Parse with json.loads() when executing the function.
    """

    name: str
    arguments: str

    def __post_init__(self) -> None:
        """Validate tool call function.

        Ensures function name and arguments are non-empty.

        Raises:
            ValueError: If name is empty/whitespace or arguments is empty.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Function name cannot be empty")
        if not self.arguments:
            raise ValueError("Function arguments cannot be empty")


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Tool call made by the model.

    Represents a function call request generated by the model during
    generation. Corresponds to POML's <tool-request> element and OpenAI's
    tool_calls format.

    Attributes:
        id: Unique tool call identifier. Used to correlate tool responses
            back to the original call. Must be non-empty.
        type: Tool call type identifier. Always "function" for function
            calling. Reserved for future extensibility.
        function: Function call details containing name and arguments.

    Raises:
        ValueError: If id is empty or contains only whitespace.

    Note:
        When a model makes a tool call, it generates one or more ToolCall
        objects. The application should execute the function and return
        results in a message with role="tool" and matching tool_call_id.
    """

    id: str
    function: ToolCallFunction
    type: Literal["function"] = "function"

    def __post_init__(self) -> None:
        """Validate tool call.

        Ensures tool call ID is non-empty.

        Raises:
            ValueError: If id is empty or contains only whitespace.
        """
        if not self.id or not self.id.strip():
            raise ValueError("Tool call ID cannot be empty")


# ============================================================================
# Generation Options and Requests
# ============================================================================


@dataclass(slots=True, frozen=True)
class GenerationOptions:
    """Options for text generation.

    Immutable configuration object for generation parameters with business
    rules enforcing valid ranges. All parameters are optional with sensible
    defaults optimized for general-purpose text generation.

    Attributes:
        temperature: Sampling temperature in range [0.0, 2.0]. Controls randomness:
            - 0.0: Deterministic (always picks most likely token)
            - 1.0: Balanced randomness
            - 2.0: Maximum randomness
            Default: 0.2 (slightly deterministic).
        top_p: Nucleus sampling parameter in range [0.0, 1.0]. Probability mass
            threshold for token selection. Lower values = more focused sampling.
            Default: 0.9.
        top_k: Top-k sampling parameter. Number of highest-probability tokens
            to consider. Must be >= 1. Default: 40.
        repeat_penalty: Penalty multiplier for repeated tokens. Values > 1.0
            reduce repetition. Default: 1.1.
        max_tokens: Maximum tokens to generate. None means no limit (generation
            continues until stop sequence or model limit). Must be >= 1 if set.
        seed: Random seed for reproducibility. None means random seed each time.
            Same seed + same prompt = deterministic output.
        stop: List of stop sequences. Generation stops immediately when any
            sequence is encountered. Sequences are matched exactly.

    Raises:
        ValueError: If any parameter is outside valid range.

    Note:
        Temperature and top_p are mutually exclusive sampling strategies.
        Typically only one is used, but both can be set for fine-grained control.
    """

    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate generation options.

        Enforces business rules for all generation parameters, ensuring
        values are within acceptable ranges.

        Raises:
            ValueError: If any parameter is outside valid range:
                - temperature not in [0.0, 2.0]
                - top_p not in [0.0, 1.0]
                - top_k < 1
                - max_tokens < 1 (if not None)
        """
        if not 0.0 <= self.temperature <= TEMPERATURE_MAX:
            raise ValueError(
                f"Temperature must be between 0.0 and {TEMPERATURE_MAX}, got {self.temperature}"
            )
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"Top-p must be between 0.0 and 1.0, got {self.top_p}")
        if self.top_k < 1:
            raise ValueError(f"Top-k must be >= 1, got {self.top_k}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"Max tokens must be >= 1, got {self.max_tokens}")


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """Domain entity for text generation requests.

    Pure domain model representing a single-prompt text generation request
    with business rules and invariants. No I/O or framework dependencies.

    This entity is used for simple text generation where a single prompt
    produces a single response. For multi-turn conversations, use ChatRequest.

    Attributes:
        prompt: Text prompt for generation. Must not be empty or whitespace-only.
            Validated by Prompt value object.
        model: Model name identifier. None uses service default model.
            Should be a text-generation capable model (not VLM).
        system: System message to set model behavior/instructions. Optional.
            Provides context and instructions that persist across the generation.
        options: Generation options (temperature, top_p, etc.). None uses model
            defaults. See GenerationOptions for parameter details.
        format: Output format specification. Controls response structure:
            - "json": Forces JSON output (JSON mode)
            - dict: JSON schema for structured output
            - None: Default text output (no format constraints)
        tools: List of tools/functions the model can call. POML compatible.
            None means no tool calling. Tools enable function calling capabilities.

    Raises:
        ValueError: If prompt is empty, whitespace-only, or exceeds max length.

    Note:
        The format parameter enables structured output. When set to "json" or
        a schema dict, the model is constrained to produce valid JSON matching
        the specification. This is useful for extracting structured data.
    """

    prompt: Prompt
    model: ModelName | None = None
    system: SystemMessage | None = None
    options: GenerationOptions | None = None
    format: str | dict[str, object] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate generation request.

        Ensures prompt is non-empty and within length limits.

        Raises:
            ValueError: If prompt is empty, whitespace-only, or exceeds
                PROMPT_MAX_LENGTH characters.
        """
        if not self.prompt.value.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        if len(self.prompt.value) > PROMPT_MAX_LENGTH:
            raise ValueError(
                f"Prompt is too long. Maximum length is {PROMPT_MAX_LENGTH:,} characters"
            )


@dataclass(slots=True, frozen=True)
class ChatMessage:
    """A chat message with role and text content (native Ollama format).

    Pure domain entity for chat messages with tool calling support. Used in
    ChatRequest for text-only conversations. For VLM conversations with images,
    use VLMMessage and VLMRequest instead.

    Message Roles:
        - "user": Human user input
        - "assistant": Model response (may include tool_calls)
        - "system": System instructions (typically first message)
        - "tool": Function execution results (requires tool_call_id)

    Attributes:
        role: Message role identifier. Must be one of VALID_ROLES.
        content: Text content of the message. Optional if tool_calls present.
            For tool role, contains function execution result.
        tool_calls: Tool calls made by assistant. Only valid for role="assistant".
            POML compatible. None if no tool calls.
        tool_call_id: Tool call ID this message responds to. Required for
            role="tool". Must match a ToolCall.id from a previous assistant message.

    Raises:
        ValueError: If role is invalid, neither content nor tool_calls present,
            tool role missing tool_call_id, or content is empty when provided.

    Note:
        Messages must have either content or tool_calls (or both). Tool messages
        (role="tool") represent function execution results and must reference
        the original tool call via tool_call_id.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate chat message.

        Enforces business rules for message structure and role-specific
        requirements.

        Raises:
            ValueError: If:
                - role is not in VALID_ROLES
                - neither content nor tool_calls is present
                - role="tool" but tool_call_id is missing
                - content is provided but empty (unless tool_calls present)
        """
        if self.role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{self.role}'. Must be 'user', 'assistant', 'system', or 'tool'")

        # Validate that either content or tool_calls is present
        if self.content is None and not self.tool_calls:
            raise ValueError("Message must have either content or tool_calls")

        # Validate tool messages have tool_call_id
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool messages must have tool_call_id")

        # Validate content is not empty if provided (unless tool_calls present)
        if self.content is not None and (not self.content or not self.content.strip()) and not self.tool_calls:
            raise ValueError("Message content cannot be empty string")


@dataclass(slots=True, frozen=True)
class ChatRequest:
    """Domain entity for chat completion requests.

    Pure domain model representing a multi-turn conversation request with
    business rules and invariants. No I/O or framework dependencies.

    Used for text-only conversations. For vision-language model requests
    with images, use VLMRequest instead.

    Attributes:
        messages: Sequence of chat messages forming the conversation history.
            Must not be empty. Messages are processed in order.
        model: Model name identifier. None uses service default text model.
            Should be a text-generation capable model (not VLM).
        options: Generation options (temperature, top_p, etc.). None uses model
            defaults. See GenerationOptions for parameter details.
        format: Output format specification. Controls response structure:
            - "json": Forces JSON output (JSON mode)
            - dict: JSON schema for structured output
            - None: Default text output (no format constraints)
        tools: List of tools/functions the model can call. POML compatible.
            None means no tool calling. Tools enable function calling capabilities.

    Raises:
        ValueError: If messages list is empty or total content exceeds
            MAX_TOTAL_MESSAGE_CHARS.

    Note:
        Messages form a conversation history. Typically starts with a system
        message, followed by alternating user/assistant messages. Tool calls
        from assistant messages are followed by tool messages with results.
    """

    messages: tuple[ChatMessage, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None
    format: str | dict[str, Any] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate chat request.

        Ensures messages list is non-empty and total content length is within
        limits.

        Raises:
            ValueError: If:
                - messages list is empty
                - total character count across all messages exceeds
                  MAX_TOTAL_MESSAGE_CHARS
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")
        total_length = sum(
            len(msg.content) if msg.content else 0 for msg in self.messages
        )
        if total_length > MAX_TOTAL_MESSAGE_CHARS:
            raise ValueError("Total message content is too long. Maximum length is 1,000,000 characters")


@dataclass(slots=True, frozen=True)
class VLMMessage:
    """Message for VLM requests with tool calling support (native Ollama format).

    Pure domain entity for vision-language model messages. Uses native Ollama
    format where images are passed separately from message text content, unlike
    OpenAI format where images are embedded in content.

    This entity extends ChatMessage with image support. Images are base64-encoded
    data URLs attached to messages, typically user messages.

    Attributes:
        role: Message role identifier. Must be one of VALID_ROLES.
        content: Text content of the message. Optional if tool_calls present.
            For tool role, contains function execution result.
        images: Tuple of base64-encoded image data URLs. Format:
            "data:image/{format};base64,{base64_data}". None if no images.
            Typically attached to user messages.
        tool_calls: Tool calls made by assistant. Only valid for role="assistant".
            POML compatible. None if no tool calls.
        tool_call_id: Tool call ID this message responds to. Required for
            role="tool". Must match a ToolCall.id from a previous assistant message.

    Raises:
        ValueError: If role is invalid, neither content nor tool_calls present,
            tool role missing tool_call_id, or content is empty when provided.

    Note:
        Images are stored as data URLs (data:image/...;base64,...). The VLM
        request must contain at least one message with images. Images are
        typically attached to user messages asking questions about images.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | None = None
    images: tuple[str, ...] | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate VLM message.

        Enforces business rules for message structure and role-specific
        requirements.

        Raises:
            ValueError: If:
                - role is not in VALID_ROLES
                - neither content nor tool_calls is present
                - role="tool" but tool_call_id is missing
                - content is provided but empty (unless tool_calls present)
        """
        if self.role not in ("user", "assistant", "system", "tool"):
            raise ValueError(f"Invalid role '{self.role}'. Must be 'user', 'assistant', 'system', or 'tool'")

        # Validate that either content or tool_calls is present
        if self.content is None and not self.tool_calls:
            raise ValueError("Message must have either content or tool_calls")

        # Validate tool messages have tool_call_id
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool messages must have tool_call_id")

        # Validate content is not empty if provided (unless tool_calls present)
        if self.content is not None and (not self.content or not self.content.strip()) and not self.tool_calls:
            raise ValueError("Message content cannot be empty string")


@dataclass(slots=True, frozen=True)
class VLMRequest:
    """Domain entity for vision-language model requests.

    Pure domain model representing a vision-language model request with images
    and text. Uses native Ollama format where images are attached to messages
    separately from text content.

    This entity is distinct from ChatRequest to enable dedicated VLM processing,
    image compression, and specialized handling for multimodal inputs.

    Attributes:
        messages: Sequence of VLM messages forming the conversation. Must not
            be empty. At least one message must contain images.
        model: Model name identifier. None uses service default VLM model.
            Should be a VLM-capable model (e.g., qwen3-vl:8b-instruct-q4_K_M).
        options: Generation options (temperature, top_p, etc.). None uses model
            defaults. See GenerationOptions for parameter details.
        image_compression: Whether to compress images before sending to model.
            Default: True. Compression reduces bandwidth and processing time.
        max_dimension: Maximum image dimension in pixels for resizing.
            Range: [MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION]. Default: 2667.
            Larger images are resized to fit within this dimension.
        format: Output format specification. Controls response structure:
            - "json": Forces JSON output (JSON mode)
            - dict: JSON schema for structured output
            - None: Default text output (no format constraints)
        tools: List of tools/functions the model can call. POML compatible.
            None means no tool calling. Tools enable function calling capabilities.

    Raises:
        ValueError: If:
            - messages list is empty
            - no message contains images
            - max_dimension is outside valid range

    Note:
        VLM requests require at least one image. Use ChatRequest for text-only
        conversations. Images are base64-encoded data URLs attached to VLMMessage
        objects. The image_compression flag enables automatic optimization before
        sending to the model.
    """

    messages: tuple[VLMMessage, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None
    image_compression: bool = True
    max_dimension: int = 2667
    format: str | dict[str, Any] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate VLM request.

        Enforces business rules: non-empty messages, at least one image,
        and valid max_dimension range.

        Raises:
            ValueError: If:
                - messages list is empty
                - no message contains images (VLM requires images)
                - max_dimension is outside [MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION]
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")

        # Validate that at least one message contains an image
        if not any(msg.images for msg in self.messages):
            raise ValueError(
                "VLM request must contain at least one image. "
                "Use ChatRequest for text-only requests."
            )

        # Validate max_dimension
        if not (MIN_IMAGE_DIMENSION <= self.max_dimension <= MAX_IMAGE_DIMENSION):
            raise ValueError(
                f"max_dimension must be between {MIN_IMAGE_DIMENSION} and {MAX_IMAGE_DIMENSION}"
            )


# ============================================================================
# OpenAI-Compatible VLM Domain Entities (for Docling and other OpenAI-compatible clients)
# ============================================================================


@dataclass(slots=True, frozen=True)
class ImageContent:
    """Image content part for OpenAI-compatible multimodal messages.

    Represents an image embedded in a multimodal message following OpenAI's
    content format. Images are base64-encoded data URLs compatible with
    OpenAI's vision API.

    This entity is used in ChatMessageOpenAI for OpenAI-compatible VLM requests.
    Images are embedded directly in message content rather than passed separately.

    Attributes:
        type: Content type identifier. Always "image_url" for images.
        image_url: Base64-encoded image data URL. Format:
            "data:image/{format};base64,{base64_data}". Must be non-empty.

    Raises:
        ValueError: If image_url is empty, doesn't start with "data:image/",
            or is missing the ";base64," separator.

    Note:
        This format matches OpenAI's vision API where images are embedded in
        message content arrays alongside text parts. The data URL format is
        required for compatibility.
    """

    type: Literal["image_url"] = "image_url"
    image_url: str = ""

    def __post_init__(self) -> None:
        """Validate image content.

        Ensures image_url is a valid base64 data URL.

        Raises:
            ValueError: If:
                - image_url is empty
                - image_url doesn't start with "data:image/"
                - image_url is missing ";base64," separator
        """
        if not self.image_url:
            raise ValueError("Image URL cannot be empty")
        if not self.image_url.startswith("data:image/"):
            raise ValueError("Image URL must start with 'data:image/'")
        if ";base64," not in self.image_url:
            raise ValueError("Image URL must contain ';base64,' separator")


@dataclass(slots=True, frozen=True)
class TextContent:
    """Text content part for OpenAI-compatible multimodal messages.

    Represents text embedded in a multimodal message following OpenAI's
    content format. Used alongside ImageContent in ChatMessageOpenAI for
    multimodal conversations.

    Attributes:
        type: Content type identifier. Always "text" for text content.
        text: Text content string. Must be non-empty after stripping.

    Raises:
        ValueError: If text is empty or contains only whitespace.

    Note:
        This format matches OpenAI's vision API where text and images are
        combined in message content arrays. TextContent and ImageContent
        can be mixed in a single message's content tuple.
    """

    type: Literal["text"] = "text"
    text: str = ""

    def __post_init__(self) -> None:
        """Validate text content.

        Ensures text is non-empty and non-whitespace.

        Raises:
            ValueError: If text is empty or contains only whitespace.
        """
        if not self.text or not self.text.strip():
            raise ValueError("Text content cannot be empty")


@dataclass(slots=True, frozen=True)
class ChatMessageOpenAI:
    """OpenAI-compatible chat message with multimodal content support.

    Supports both simple string content and multimodal content (text + images)
    following OpenAI's message format. Used for OpenAI-compatible VLM requests
    from clients like Docling that expect OpenAI's API structure.

    Content Formats:
        - String: Simple text-only message (e.g., "What's in this image?")
        - Tuple: Multimodal content with TextContent and ImageContent parts
          (e.g., [TextContent("Describe"), ImageContent("data:image/...")])

    Attributes:
        role: Message role identifier. Must be "user", "assistant", or "system".
            Note: OpenAI format doesn't support "tool" role in messages.
        content: Message content. Either:
            - str: Text-only content (must be non-empty)
            - tuple: Multimodal content parts (TextContent, ImageContent)
              (must be non-empty)

    Raises:
        ValueError: If:
            - role is not "user", "assistant", or "system"
            - content is empty string or empty tuple
            - content type is neither str nor tuple

    Note:
        This entity is converted internally to native Ollama format (VLMMessage)
        for processing. The OpenAI format embeds images in content, while Ollama
        format attaches images separately to messages.
    """

    role: Literal["user", "assistant", "system"]
    content: str | tuple[ImageContent | TextContent, ...]

    def __post_init__(self) -> None:
        """Validate OpenAI-compatible message.

        Ensures role is valid and content is non-empty.

        Raises:
            ValueError: If role is invalid, content is empty, or content type
                is invalid.
        """
        if self.role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role '{self.role}'. Must be 'user', 'assistant', or 'system'")

        if isinstance(self.content, str):
            if not self.content or not self.content.strip():
                raise ValueError("Text content cannot be empty")
        elif isinstance(self.content, tuple):
            if not self.content:
                raise ValueError("Content parts list cannot be empty")
        else:
            raise ValueError("Content must be either string or tuple of content parts")


@dataclass(slots=True, frozen=True)
class VLMRequestOpenAI:
    """OpenAI-compatible VLM request domain entity.

    Pure domain model representing a vision-language model request using
    OpenAI's multimodal message format. Images are embedded in message content
    arrays rather than passed separately.

    This entity is converted internally to native Ollama format (VLMRequest)
    for processing. Used for OpenAI-compatible clients like Docling that expect
    OpenAI's API structure.

    Attributes:
        messages: Sequence of OpenAI-compatible chat messages with multimodal
            content. Must not be empty. At least one message must contain images
            in its content array.
        model: Model name identifier. None uses service default VLM model.
            Should be a VLM-capable model (e.g., qwen3-vl:8b-instruct-q4_K_M).
        options: Generation options (temperature, top_p, etc.). None uses model
            defaults. See GenerationOptions for parameter details.
        image_compression: Whether to compress images before sending to model.
            Default: True. Compression reduces bandwidth and processing time.
        max_dimension: Maximum image dimension in pixels for resizing.
            Range: [MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION]. Default: 2667.
            Larger images are resized to fit within this dimension.
        format: Output format specification. Controls response structure:
            - "json": Forces JSON output (JSON mode)
            - dict: JSON schema for structured output
            - None: Default text output (no format constraints)
        tools: List of tools/functions the model can call. POML compatible.
            None means no tool calling. Tools enable function calling capabilities.

    Raises:
        ValueError: If:
            - messages list is empty
            - no message contains images (VLM requires images)
            - max_dimension is outside valid range

    Note:
        This format embeds images in message content arrays (OpenAI style),
        while native Ollama format attaches images separately to messages.
        The mapper converts between these formats automatically.
    """

    messages: tuple[ChatMessageOpenAI, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None
    image_compression: bool = True
    max_dimension: int = 2667
    format: str | dict[str, Any] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate OpenAI-compatible VLM request.

        Enforces business rules: non-empty messages, at least one image in
        content arrays, and valid max_dimension range.

        Raises:
            ValueError: If:
                - messages list is empty
                - no message contains ImageContent in its content array
                - max_dimension is outside [MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION]
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")

        # Validate that at least one message contains an image
        has_image = False
        for msg in self.messages:
            if isinstance(msg.content, tuple):
                for part in msg.content:
                    if isinstance(part, ImageContent):
                        has_image = True
                        break
            if has_image:
                break

        if not has_image:
            raise ValueError(
                "VLM request must contain at least one image. "
                "Use ChatRequest for text-only requests."
            )

        # Validate max_dimension
        if not (MIN_IMAGE_DIMENSION <= self.max_dimension <= MAX_IMAGE_DIMENSION):
            raise ValueError(
                f"max_dimension must be between {MIN_IMAGE_DIMENSION} and {MAX_IMAGE_DIMENSION}"
            )
