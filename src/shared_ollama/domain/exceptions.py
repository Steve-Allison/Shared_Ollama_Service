"""Domain exceptions for the Shared Ollama Service.

This module defines pure domain exceptions with no framework dependencies.
These exceptions represent business rule violations and domain errors that
can occur during domain entity validation and business logic execution.

Design Principles:
    - Framework-agnostic: No FastAPI, Pydantic, or other framework deps
    - Domain-focused: Represent business rule violations, not technical errors
    - Hierarchical: All domain exceptions inherit from DomainError
    - Descriptive: Exception names clearly indicate the violation type

Exception Hierarchy:
    - DomainError: Base exception for all domain errors
    - InvalidModelError: Model name or availability issues
    - InvalidPromptError: Prompt validation failures
    - InvalidRequestError: General request validation failures
"""


class DomainError(Exception):
    """Base exception for all domain errors.

    All domain-specific exceptions inherit from this class, enabling
    catch-all exception handling for domain violations.

    This exception should not be raised directly. Use specific subclasses
    like InvalidModelError or InvalidPromptError instead.

    Note:
        Catching DomainError will catch all domain exceptions, useful for
        converting domain errors to API errors in the application layer.
    """


class InvalidModelError(DomainError):
    """Raised when a model name is invalid or model is not available.

    This exception indicates a business rule violation related to model
    selection or availability. Common causes:
        - Model name doesn't exist
        - Model is not available in the current hardware profile
        - Model name format is invalid

    Note:
        This is a domain error, not a technical error. It represents a
        business rule violation that should be communicated to the user.
    """


class InvalidPromptError(DomainError):
    """Raised when a prompt is invalid (empty, too long, etc.).

    This exception indicates a business rule violation related to prompt
    validation. Common causes:
        - Prompt is empty or whitespace-only
        - Prompt exceeds maximum length (PROMPT_MAX_LENGTH)
        - Prompt contains invalid characters (if validation added)

    Note:
        This exception is typically raised by Prompt value object or
        GenerationRequest validation logic.
    """


class InvalidRequestError(DomainError):
    """Raised when a request violates business rules.

    This exception indicates a general business rule violation in a request.
    Used for validation failures that don't fit more specific exception
    types like InvalidModelError or InvalidPromptError.

    Common causes:
        - Request structure violates domain invariants
        - Request parameters are outside valid ranges
        - Request conflicts with business rules

    Note:
        This is a catch-all for domain validation errors. More specific
        exceptions should be used when appropriate (e.g., InvalidModelError).
    """
