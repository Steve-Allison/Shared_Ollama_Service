"""Domain exceptions for the Shared Ollama Service.

Pure domain exceptions with no framework dependencies. These exceptions
represent business rule violations and domain errors.
"""


class DomainError(Exception):
    """Base exception for all domain errors.

    All domain-specific exceptions inherit from this class.
    """


class InvalidModelError(DomainError):
    """Raised when a model name is invalid or model is not available."""


class InvalidPromptError(DomainError):
    """Raised when a prompt is invalid (empty, too long, etc.)."""


class InvalidRequestError(DomainError):
    """Raised when a request violates business rules."""

