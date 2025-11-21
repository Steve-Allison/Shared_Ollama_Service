"""Value objects for the Shared Ollama Service.

This module defines immutable value objects representing domain concepts
with no identity. Value objects encapsulate validation and business rules
for primitive domain values.

Design Principles:
    - Immutability: All value objects are frozen dataclasses (slots=True)
    - Validation: Business rules enforced in __post_init__ methods
    - No identity: Value objects are compared by value, not reference
    - Self-contained: Each value object validates its own constraints

Key Value Objects:
    - ModelName: Validated model identifier
    - Prompt: Validated text prompt with length limits
    - SystemMessage: System instruction text (allows empty strings)
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_MAX_LENGTH = 1_000_000
"""Maximum character length for prompts (inclusive)."""


@dataclass(slots=True, frozen=True)
class ModelName:
    """Value object representing a model name.

    Validates and encapsulates model name with business rules. Model names
    must be non-empty and typically follow Ollama's naming convention
    (e.g., "qwen3-vl:8b-instruct-q4_K_M").

    Attributes:
        value: Model name string. Must not be empty or whitespace-only.
            Should match Ollama model naming patterns.

    Raises:
        ValueError: If model name is empty or contains only whitespace.

    Note:
        This value object ensures model names are always valid before use
        in domain entities. Empty or whitespace-only names are rejected.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate model name.

        Ensures model name is non-empty and non-whitespace.

        Raises:
            ValueError: If model name is empty or contains only whitespace.
        """
        if not self.value or not self.value.strip():
            raise ValueError("Model name cannot be empty")


@dataclass(slots=True, frozen=True)
class Prompt:
    """Value object representing a text prompt.

    Validates and encapsulates prompt text with business rules including
    length limits. Used in GenerationRequest to ensure prompts are valid
    before processing.

    Attributes:
        value: Prompt text string. Must not be empty or whitespace-only.
            Must not exceed PROMPT_MAX_LENGTH characters.

    Raises:
        ValueError: If prompt is empty, whitespace-only, or exceeds
            PROMPT_MAX_LENGTH characters.

    Note:
        Prompts are validated for both emptiness and length. Very long
        prompts may be rejected to prevent resource exhaustion.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate prompt.

        Ensures prompt is non-empty and within length limits.

        Raises:
            ValueError: If:
                - prompt is empty or contains only whitespace
                - prompt length exceeds PROMPT_MAX_LENGTH characters
        """
        if not self.value or not self.value.strip():
            raise ValueError("Prompt cannot be empty")
        if len(self.value) > PROMPT_MAX_LENGTH:
            raise ValueError("Prompt is too long. Maximum length is 1,000,000 characters")


@dataclass(slots=True, frozen=True)
class SystemMessage:
    """Value object representing a system message.

    Encapsulates system message text used to set model behavior and
    instructions. System messages are optional, so empty strings are
    explicitly allowed (unlike Prompt which rejects empty values).

    Attributes:
        value: System message text. May be empty string. Empty system
            messages are valid and indicate no system instructions.

    Note:
        Unlike Prompt, SystemMessage allows empty strings because system
        messages are optional. An empty system message means the model
        uses its default behavior without custom instructions.
    """

    value: str
