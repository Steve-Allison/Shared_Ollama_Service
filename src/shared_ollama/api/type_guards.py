"""Type guards for runtime type checking.

This module provides type guard functions to replace assert isinstance()
patterns with proper type narrowing that works even when assertions are disabled.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from typing import Any


def is_dict_result(result: dict[str, Any] | AsyncIterator[dict[str, Any]]) -> TypeGuard[dict[str, Any]]:
    """Type guard to check if result is a dict (non-streaming).

    Args:
        result: Result from use case execute() method.

    Returns:
        True if result is a dict, False if it's an AsyncIterator.
    """
    return isinstance(result, dict)


def is_streaming_result(result: dict[str, Any] | AsyncIterator[dict[str, Any]]) -> TypeGuard[AsyncIterator[dict[str, Any]]]:
    """Type guard to check if result is an AsyncIterator (streaming).

    Args:
        result: Result from use case execute() method.

    Returns:
        True if result is an AsyncIterator, False if it's a dict.
    """
    return isinstance(result, AsyncIterator)
