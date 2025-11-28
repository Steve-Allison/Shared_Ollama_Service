"""High-level validation helpers used by API routes."""

from __future__ import annotations

import json
import math
from typing import Iterable, Sequence

from fastapi import HTTPException, status

from shared_ollama.api.limits import MAX_PROMPT_CHARS, MAX_PROMPT_TOKENS
from shared_ollama.api.models import (
    ChatMessage,
    ChatMessageOpenAI,
    ContentPart,
    TextContentPart,
    VLMMessage,
)

ChatMessageTypes = Sequence[ChatMessage | VLMMessage]
OpenAIMessageTypes = Sequence[ChatMessageOpenAI]


def _collect_text_from_content(content: object) -> str:
    """Return textual content from either a plain string or ContentPart list."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        buffer: list[str] = []
        for part in content:  # type: ignore[assignment]
            if isinstance(part, TextContentPart):
                buffer.append(part.text)
        return "\n".join(buffer)
    return ""


def _estimate_prompt_tokens(char_count: int) -> int:
    if char_count <= 0:
        return 0
    return int(math.ceil(char_count / 4))


def _count_tool_call_text(tool_calls: object) -> str:
    if not tool_calls:
        return ""
    try:
        return json.dumps(
            [tc.model_dump(exclude_none=True) for tc in tool_calls],  # type: ignore[attr-defined]
            ensure_ascii=False,
        )
    except Exception:
        return ""


def _gather_char_count(messages: Sequence[object]) -> tuple[int, int]:
    total_chars = 0
    counted_messages = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        total_chars += len(_collect_text_from_content(content))
        total_chars += len(_count_tool_call_text(getattr(msg, "tool_calls", None)))
        counted_messages += 1
    return total_chars, counted_messages


def enforce_native_prompt_limit(messages: ChatMessageTypes, *, request_label: str) -> None:
    """Ensure native (Ollama-style) chat/VLM messages stay within prompt limit."""

    char_count, _ = _gather_char_count(messages)
    if char_count <= MAX_PROMPT_CHARS:
        return

    estimated_tokens = _estimate_prompt_tokens(char_count)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "code": "prompt_too_large",
            "message": (
                f"{request_label} prompt is approximately {estimated_tokens:,} tokens "
                f"(limit {MAX_PROMPT_TOKENS:,}). Trim retrieved context or summarize history "
                "before retrying."
            ),
            "estimated_tokens": estimated_tokens,
            "limit_tokens": MAX_PROMPT_TOKENS,
        },
    )


def enforce_openai_prompt_limit(messages: OpenAIMessageTypes, *, request_label: str) -> None:
    """Ensure OpenAI-compatible chat/VLM messages stay within prompt limit."""

    char_count, _ = _gather_char_count(messages)
    if char_count <= MAX_PROMPT_CHARS:
        return

    estimated_tokens = _estimate_prompt_tokens(char_count)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "code": "prompt_too_large",
            "message": (
                f"{request_label} prompt is approximately {estimated_tokens:,} tokens "
                f"(limit {MAX_PROMPT_TOKENS:,}). Trim retrieved context or summarize history "
                "before retrying."
            ),
            "estimated_tokens": estimated_tokens,
            "limit_tokens": MAX_PROMPT_TOKENS,
        },
    )


def enforce_text_prompt_limit(
    prompt: str | None,
    *,
    request_label: str,
    extra_chunks: Sequence[str | None] | None = None,
) -> None:
    """Ensure single-string prompts stay within token limits."""

    total_chars = len(prompt or "")
    if extra_chunks:
        for chunk in extra_chunks:
            if chunk:
                total_chars += len(chunk)

    if total_chars <= MAX_PROMPT_CHARS:
        return

    estimated_tokens = _estimate_prompt_tokens(total_chars)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "code": "prompt_too_large",
            "message": (
                f"{request_label} prompt is approximately {estimated_tokens:,} tokens "
                f"(limit {MAX_PROMPT_TOKENS:,}). Trim the prompt or system message and retry."
            ),
            "estimated_tokens": estimated_tokens,
            "limit_tokens": MAX_PROMPT_TOKENS,
        },
    )


__all__ = [
    "enforce_native_prompt_limit",
    "enforce_openai_prompt_limit",
    "enforce_text_prompt_limit",
]

