"""Domain entities for the Shared Ollama Service.

Pure domain models representing core business concepts with no framework
or infrastructure dependencies. These entities contain business rules and
invariants.

All entities are immutable dataclasses with slots=True for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage


class Model(StrEnum):
    """Available Ollama models.

    Predefined model identifiers for common Ollama models. Values are
    the model names as recognized by the Ollama service.
    """

    QWEN25_VL_7B = "qwen2.5vl:7b"  # Primary: 7B params, vision-language model
    QWEN25_14B = "qwen2.5:14b"  # Secondary: 14.8B params
    GRANITE_4_SMALL = (
        "granite4:small-h"  # Granite 4.0 Small: 8B params, instruction-tuned
    )


@dataclass(slots=True, frozen=True)
class ModelInfo:
    """Information about an available model.

    Pure domain entity representing model metadata. No I/O or framework
    dependencies.

    Attributes:
        name: Model name identifier.
        size: Model size in bytes, if available.
        modified_at: ISO timestamp of last modification, if available.
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

    Compatible with both POML <tool-definition> and OpenAI function calling.

    Attributes:
        name: Function name (must be non-empty).
        description: Function description (optional).
        parameters: JSON schema for function parameters.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate tool function.

        Raises:
            ValueError: If name is empty.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Function name cannot be empty")


@dataclass(slots=True, frozen=True)
class Tool:
    """Tool definition for function calling.

    Attributes:
        type: Tool type. Always "function".
        function: Function definition.
    """

    function: ToolFunction
    type: Literal["function"] = "function"


@dataclass(slots=True, frozen=True)
class ToolCallFunction:
    """Function call details in a tool call.

    Attributes:
        name: Function name being called.
        arguments: JSON string of function arguments.
    """

    name: str
    arguments: str

    def __post_init__(self) -> None:
        """Validate tool call function.

        Raises:
            ValueError: If name or arguments are empty.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Function name cannot be empty")
        if not self.arguments:
            raise ValueError("Function arguments cannot be empty")


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Tool call made by the model.

    Corresponds to POML's <tool-request> element.

    Attributes:
        id: Unique tool call ID.
        type: Tool call type. Always "function".
        function: Function call details.
    """

    id: str
    function: ToolCallFunction
    type: Literal["function"] = "function"

    def __post_init__(self) -> None:
        """Validate tool call.

        Raises:
            ValueError: If id is empty.
        """
        if not self.id or not self.id.strip():
            raise ValueError("Tool call ID cannot be empty")


# ============================================================================
# Generation Options and Requests
# ============================================================================


@dataclass(slots=True, frozen=True)
class GenerationOptions:
    """Options for text generation.

    Immutable configuration object for generation parameters. Contains
    business rules for valid ranges.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Lower values make output
            more deterministic (default: 0.2).
        top_p: Nucleus sampling parameter (0.0-1.0) (default: 0.9).
        top_k: Top-k sampling parameter. Number of tokens to consider
            (default: 40).
        repeat_penalty: Penalty for repetition (default: 1.1).
        max_tokens: Maximum tokens to generate. None means no limit.
        seed: Random seed for reproducibility. None means random.
        stop: List of stop sequences. Generation stops when any sequence
            is encountered.
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

        Raises:
            ValueError: If any parameter is outside valid range.
        """
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"Top-p must be between 0.0 and 1.0, got {self.top_p}")
        if self.top_k < 1:
            raise ValueError(f"Top-k must be >= 1, got {self.top_k}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"Max tokens must be >= 1, got {self.max_tokens}")


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """Domain entity for text generation requests.

    Pure domain model representing a generation request with business rules
    and invariants. No I/O or framework dependencies.

    Attributes:
        prompt: Text prompt for generation. Must not be empty.
        model: Model name. Optional, defaults to service default.
        system: System message to set model behavior. Optional.
        options: Generation options. Optional.
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        tools: List of tools/functions the model can call (POML compatible).
    """

    prompt: Prompt
    model: ModelName | None = None
    system: SystemMessage | None = None
    options: GenerationOptions | None = None
    format: str | dict[str, object] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate generation request.

        Raises:
            InvalidPromptError: If prompt is invalid.
        """
        if not self.prompt.value.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        if len(self.prompt.value) > 1_000_000:  # 1M characters
            raise ValueError("Prompt is too long. Maximum length is 1,000,000 characters")


@dataclass(slots=True, frozen=True)
class ChatMessage:
    """A chat message with role and text content (native Ollama format).

    Pure domain entity for chat messages with tool calling support.
    For VLM with images, use VLMRequest and VLMMessage.

    Attributes:
        role: Message role. Must be "user", "assistant", "system", or "tool".
        content: Text content of the message (optional if tool_calls present).
        tool_calls: Tool calls made by assistant (POML compatible).
        tool_call_id: Tool call ID this message responds to (for role="tool").
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate chat message.

        Raises:
            ValueError: If role is invalid or neither content nor tool_calls present.
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
class ChatRequest:
    """Domain entity for chat completion requests.

    Pure domain model representing a chat request with business rules
    and invariants. No I/O or framework dependencies.

    Attributes:
        messages: List of chat messages. Must not be empty.
        model: Model name. Optional, defaults to service default.
        options: Generation options. Optional.
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        tools: List of tools/functions the model can call (POML compatible).
    """

    messages: tuple[ChatMessage, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None
    format: str | dict[str, Any] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate chat request.

        Raises:
            ValueError: If messages are invalid.
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")
        total_length = sum(
            len(msg.content) if msg.content else 0 for msg in self.messages
        )
        if total_length > 1_000_000:  # 1M characters
            raise ValueError("Total message content is too long. Maximum length is 1,000,000 characters")


@dataclass(slots=True, frozen=True)
class VLMMessage:
    """Message for VLM requests with tool calling support (native Ollama format).

    Pure domain entity for VLM messages. Uses native Ollama format where
    images are passed separately from message text content.

    Attributes:
        role: Message role. Must be "user", "assistant", "system", or "tool".
        content: Text content of the message (optional if tool_calls present).
        tool_calls: Tool calls made by assistant (POML compatible).
        tool_call_id: Tool call ID this message responds to (for role="tool").
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate VLM message.

        Raises:
            ValueError: If role is invalid or neither content nor tool_calls present.
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

    Uses native Ollama format with separate images parameter.
    Separates VLM requests from text-only chat for dedicated processing.

    Attributes:
        messages: Text-only chat messages (native Ollama format).
        images: List of base64-encoded image data URLs.
        model: Model name (should be VLM-capable like qwen2.5vl:7b).
        options: Generation options.
        image_compression: Whether to compress images (default: True).
        max_dimension: Maximum image dimension for resizing (default: 1024).
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        tools: List of tools/functions the model can call (POML compatible).
    """

    messages: tuple[VLMMessage, ...]
    images: tuple[str, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None
    image_compression: bool = True
    max_dimension: int = 1024
    format: str | dict[str, Any] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate VLM request.

        Raises:
            ValueError: If messages/images are invalid or no images present.
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")

        # Validate images
        if not self.images:
            raise ValueError(
                "VLM request must contain at least one image. "
                "Use ChatRequest for text-only requests."
            )

        # Validate max_dimension
        if self.max_dimension < 256 or self.max_dimension > 2048:
            raise ValueError("max_dimension must be between 256 and 2048")


# ============================================================================
# OpenAI-Compatible VLM Domain Entities (for Docling and other OpenAI-compatible clients)
# ============================================================================


@dataclass(slots=True, frozen=True)
class ImageContent:
    """Image content part for OpenAI-compatible multimodal messages.

    Represents an image embedded in a multimodal message. Images are
    base64-encoded data URLs.

    Attributes:
        type: Content type. Always "image_url".
        image_url: Base64-encoded image data URL (data:image/...;base64,...).
    """

    type: Literal["image_url"] = "image_url"
    image_url: str = ""

    def __post_init__(self) -> None:
        """Validate image content.

        Raises:
            ValueError: If image_url is invalid or empty.
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

    Represents text embedded in a multimodal message.

    Attributes:
        type: Content type. Always "text".
        text: Text content string.
    """

    type: Literal["text"] = "text"
    text: str = ""

    def __post_init__(self) -> None:
        """Validate text content.

        Raises:
            ValueError: If text is empty.
        """
        if not self.text or not self.text.strip():
            raise ValueError("Text content cannot be empty")


@dataclass(slots=True, frozen=True)
class ChatMessageOpenAI:
    """OpenAI-compatible chat message with multimodal content support.

    Supports both simple string content and multimodal content (text + images).
    Used for OpenAI-compatible VLM requests.

    Attributes:
        role: Message role. Must be "user", "assistant", or "system".
        content: Either a string (text-only) or list of content parts (multimodal).
    """

    role: Literal["user", "assistant", "system"]
    content: str | tuple[ImageContent | TextContent, ...]

    def __post_init__(self) -> None:
        """Validate OpenAI-compatible message.

        Raises:
            ValueError: If role is invalid or content is empty.
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

    Uses OpenAI-compatible multimodal message format where images are embedded
    in message content. Converted internally to native Ollama format for processing.

    For Docling and other OpenAI-compatible clients.

    Attributes:
        messages: OpenAI-compatible chat messages with multimodal content.
        model: Model name (should be VLM-capable like qwen2.5vl:7b).
        options: Generation options.
        image_compression: Whether to compress images (default: True).
        max_dimension: Maximum image dimension for resizing (default: 1024).
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        tools: List of tools/functions the model can call (POML compatible).
    """

    messages: tuple[ChatMessageOpenAI, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None
    image_compression: bool = True
    max_dimension: int = 1024
    format: str | dict[str, Any] | None = None
    tools: tuple[Tool, ...] | None = None

    def __post_init__(self) -> None:
        """Validate OpenAI-compatible VLM request.

        Raises:
            ValueError: If messages are invalid or no images present.
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
        if self.max_dimension < 256 or self.max_dimension > 2048:
            raise ValueError("max_dimension must be between 256 and 2048")
