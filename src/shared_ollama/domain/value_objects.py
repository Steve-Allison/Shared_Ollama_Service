"""Value objects for the Shared Ollama Service.

Value objects are immutable objects that represent domain concepts
with no identity. They contain validation and business rules.

All value objects are immutable dataclasses with slots=True.
"""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_MAX_LENGTH = 1_000_000


@dataclass(slots=True, frozen=True)
class ModelName:
    """Value object representing a model name.

    Validates and encapsulates model name with business rules.

    Attributes:
        value: Model name string. Must not be empty.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate model name.

        Raises:
            ValueError: If model name is empty or invalid.
        """
        if not self.value or not self.value.strip():
            raise ValueError("Model name cannot be empty")


@dataclass(slots=True, frozen=True)
class Prompt:
    """Value object representing a text prompt.

    Validates and encapsulates prompt text with business rules.

    Attributes:
        value: Prompt text. Must not be empty.
    """

    value: str

    def __post_init__(self) -> None:
        """Validate prompt.

        Raises:
            ValueError: If prompt is empty or too long.
        """
        if not self.value or not self.value.strip():
            raise ValueError("Prompt cannot be empty")
        if len(self.value) > PROMPT_MAX_LENGTH:
            raise ValueError("Prompt is too long. Maximum length is 1,000,000 characters")


@dataclass(slots=True, frozen=True)
class SystemMessage:
    """Value object representing a system message.

    Encapsulates system message text. System messages are optional,
    so empty strings are allowed.

    Attributes:
        value: System message text.
    """

    value: str
