"""Domain entities for the Shared Ollama Service.

Pure domain models representing core business concepts with no framework
or infrastructure dependencies. These entities contain business rules and
invariants.

All entities are immutable dataclasses with slots=True for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from shared_ollama.domain.value_objects import ModelName, Prompt, SystemMessage


class Model(StrEnum):
    """Available Ollama models.

    Predefined model identifiers for common Ollama models. Values are
    the model names as recognized by the Ollama service.
    """

    QWEN25_VL_7B = "qwen2.5vl:7b"  # Primary: 7B params, vision-language model
    QWEN25_7B = "qwen2.5:7b"  # Standard: 7B params, text-only model
    QWEN25_14B = "qwen2.5:14b"  # Secondary: 14.8B params
    GRANITE_4_H_TINY = (
        "granite4:tiny-h"  # Granite 4.0 H Tiny: 7B total (1B active), hybrid MoE
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
    """

    prompt: Prompt
    model: ModelName | None = None
    system: SystemMessage | None = None
    options: GenerationOptions | None = None
    format: str | dict[str, object] | None = None

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
    """A chat message with role and content.

    Pure domain entity representing a single message in a conversation.
    Contains business rules for valid roles.

    Attributes:
        role: Message role. Must be "user", "assistant", or "system".
        content: Message content. Must not be empty.
    """

    role: Literal["user", "assistant", "system"]
    content: str

    def __post_init__(self) -> None:
        """Validate chat message.

        Raises:
            ValueError: If role is invalid or content is empty.
        """
        if self.role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role '{self.role}'. Must be 'user', 'assistant', or 'system'")
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")


@dataclass(slots=True, frozen=True)
class ChatRequest:
    """Domain entity for chat completion requests.

    Pure domain model representing a chat request with business rules
    and invariants. No I/O or framework dependencies.

    Attributes:
        messages: List of chat messages. Must not be empty.
        model: Model name. Optional, defaults to service default.
        options: Generation options. Optional.
    """

    messages: tuple[ChatMessage, ...]
    model: ModelName | None = None
    options: GenerationOptions | None = None

    def __post_init__(self) -> None:
        """Validate chat request.

        Raises:
            ValueError: If messages are invalid.
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")
        total_length = sum(len(msg.content) for msg in self.messages)
        if total_length > 1_000_000:  # 1M characters
            raise ValueError("Total message content is too long. Maximum length is 1,000,000 characters")

