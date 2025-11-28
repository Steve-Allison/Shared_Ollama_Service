"""Shared API guardrail constants."""

from __future__ import annotations

# Request payload limits
MAX_REQUEST_BODY_BYTES = 1_500_000  # ~1.43 MiB, leaves headroom for headers

# Prompt sizing heuristics
MAX_PROMPT_TOKENS = 4096
_AVERAGE_CHARS_PER_TOKEN = 4  # conservative heuristic for UTF-8 text
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * _AVERAGE_CHARS_PER_TOKEN

# Backpressure guidance
QUEUE_RETRY_AFTER_SECONDS = 2

__all__ = [
    "MAX_PROMPT_CHARS",
    "MAX_PROMPT_TOKENS",
    "MAX_REQUEST_BODY_BYTES",
    "QUEUE_RETRY_AFTER_SECONDS",
]

