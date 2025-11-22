"""Shared response building utilities for route handlers.

This module provides helper functions to build consistent API responses,
reducing code duplication across route handlers. All response builders
convert use case result dictionaries into Pydantic response models.

Design Principles:
    - Consistency: Uniform response structure across all endpoints
    - Type Safety: Returns validated Pydantic models
    - Data Transformation: Converts nanosecond durations to milliseconds
    - Request Context: Includes request_id and metadata in responses

Response Builders:
    - build_chat_response(): Builds ChatResponse from chat use case result
    - build_generate_response(): Builds GenerateResponse from generate use case result
    - build_vlm_response(): Builds VLMResponse from VLM use case result
    - build_streaming_response(): Builds streaming response for SSE/EventSource

Data Transformations:
    - Duration Conversion: Nanoseconds (Ollama) to milliseconds (API)
    - Rounding: All duration values rounded to 3 decimal places
    - Optional Fields: None values for optional fields (model_load_ms, etc.)
    - Warm Start Detection: model_warm_start = True if load_duration == 0
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

from fastapi.responses import JSONResponse

from shared_ollama.api.models import (
    ChatMessage,
    ChatResponse,
    GenerateResponse,
    RequestContext,
    VLMResponse,
)


def build_chat_response(
    result_dict: dict[str, Any],
    ctx: RequestContext,
) -> ChatResponse:
    """Build ChatResponse from use case result dictionary.

    Args:
        result_dict: Result dictionary from ChatUseCase.execute().
        ctx: Request context with request_id.

    Returns:
        ChatResponse model instance.
    """
    message_content = result_dict.get("message", {}).get("content", "")
    model_used = result_dict.get("model", "unknown")
    prompt_eval_count = result_dict.get("prompt_eval_count", 0)
    eval_count = result_dict.get("eval_count", 0)
    total_duration = result_dict.get("total_duration", 0)
    load_duration = result_dict.get("load_duration", 0)

    load_ms = load_duration / 1_000_000 if load_duration else 0.0
    total_ms = total_duration / 1_000_000 if total_duration else 0.0

    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=message_content,
            tool_calls=None,
            tool_call_id=None,
        ),
        model=model_used,
        request_id=ctx.request_id,
        latency_ms=0.0,  # Use case handles latency tracking internally
        model_load_ms=round(load_ms, 3) if load_ms else None,
        model_warm_start=load_ms == 0.0,
        prompt_eval_count=prompt_eval_count,
        generation_eval_count=eval_count,
        total_duration_ms=round(total_ms, 3) if total_ms else None,
    )


def build_generate_response(
    result_dict: dict[str, Any],
    ctx: RequestContext,
) -> GenerateResponse:
    """Build GenerateResponse from use case result dictionary.

    Args:
        result_dict: Result dictionary from GenerateUseCase.execute().
        ctx: Request context with request_id.

    Returns:
        GenerateResponse model instance.
    """
    text = result_dict.get("text", "")
    model_used = result_dict.get("model", "unknown")
    prompt_eval_count = result_dict.get("prompt_eval_count", 0)
    eval_count = result_dict.get("eval_count", 0)
    total_duration = result_dict.get("total_duration", 0)
    load_duration = result_dict.get("load_duration", 0)

    load_ms = load_duration / 1_000_000 if load_duration else 0.0
    total_ms = total_duration / 1_000_000 if total_duration else 0.0

    return GenerateResponse(
        text=text,
        model=model_used,
        request_id=ctx.request_id,
        latency_ms=0.0,  # Use case handles latency tracking internally
        model_load_ms=round(load_ms, 3) if load_ms else None,
        model_warm_start=load_ms == 0.0,
        prompt_eval_count=prompt_eval_count,
        generation_eval_count=eval_count,
        total_duration_ms=round(total_ms, 3) if total_ms else None,
    )


def build_vlm_response(
    result_dict: dict[str, Any],
    ctx: RequestContext,
    images_processed: int,
) -> VLMResponse:
    """Build VLMResponse from use case result dictionary.

    Args:
        result_dict: Result dictionary from VLMUseCase.execute().
        ctx: Request context with request_id.
        images_processed: Number of images processed.

    Returns:
        VLMResponse model instance.
    """
    message_content = result_dict.get("message", {}).get("content", "")
    model_used = result_dict.get("model", "unknown")
    prompt_eval_count = result_dict.get("prompt_eval_count", 0)
    eval_count = result_dict.get("eval_count", 0)
    total_duration = result_dict.get("total_duration", 0)
    load_duration = result_dict.get("load_duration", 0)
    compression_savings = result_dict.get("compression_savings_bytes")

    load_ms = load_duration / 1_000_000 if load_duration else 0.0
    total_ms = total_duration / 1_000_000 if total_duration else 0.0

    return VLMResponse(
        message=ChatMessage(
            role="assistant",
            content=message_content,
            tool_calls=None,
            tool_call_id=None,
        ),
        model=model_used,
        request_id=ctx.request_id,
        latency_ms=0.0,  # Use case handles latency tracking internally
        model_load_ms=round(load_ms, 3) if load_ms else None,
        model_warm_start=load_ms == 0.0,
        images_processed=images_processed,
        compression_savings_bytes=compression_savings,
        prompt_eval_count=prompt_eval_count,
        generation_eval_count=eval_count,
        total_duration_ms=round(total_ms, 3) if total_ms else None,
    )


def json_response(model: ChatResponse | GenerateResponse | VLMResponse) -> JSONResponse:
    """Create JSONResponse from Pydantic model.

    Uses Pydantic's model_dump() with mode='json' for efficient JSON serialization
    with proper datetime handling.

    Args:
        model: Pydantic response model.

    Returns:
        JSONResponse with model data.
    """
    return JSONResponse(content=model.model_dump(mode="json"))


def _coerce_created_timestamp(value: Any) -> int:
    """Normalize created_at values to a Unix timestamp."""

    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            try:
                # Support ISO8601 strings with optional trailing Z
                sanitized = value.replace("Z", "+00:00") if value.endswith("Z") else value
                return int(datetime.fromisoformat(sanitized).timestamp())
            except ValueError:
                pass
    return int(time.time())


def build_openai_chat_response(
    result_dict: dict[str, Any],
    ctx: RequestContext,
) -> dict[str, Any]:
    """Convert native use-case result into OpenAI chat completion schema.

    Builds a fully OpenAI-compliant chat completion response including all
    required and optional fields per the OpenAI API specification.

    Args:
        result_dict: Result dictionary from use case execute().
        ctx: Request context with request_id.

    Returns:
        OpenAI-compliant chat completion response dictionary.
    """
    message = result_dict.get("message") or {}
    model_used = result_dict.get("model", "unknown")
    prompt_eval_count = result_dict.get("prompt_eval_count", 0)
    eval_count = result_dict.get("eval_count", 0)
    created_ts = _coerce_created_timestamp(result_dict.get("created_at"))
    finish_reason = result_dict.get("finish_reason", "stop")

    # Build OpenAI-compliant message payload with all required fields
    message_payload: dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": message.get("content", ""),
        "refusal": message.get("refusal"),  # Required: null when not refusing
    }

    # Add tool_calls if present (OpenAI spec: null when not present)
    if "tool_calls" in message and message["tool_calls"] is not None:
        message_payload["tool_calls"] = message["tool_calls"]

    # Add function_call if present (deprecated but still supported)
    if "function_call" in message and message["function_call"] is not None:
        message_payload["function_call"] = message["function_call"]

    # Add annotations if present (OpenAI spec: empty array by default)
    message_payload["annotations"] = message.get("annotations", [])

    return {
        "id": ctx.request_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created_ts,
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": message_payload,
                "logprobs": None,  # Required: null unless logprobs requested
                "finish_reason": finish_reason,
            },
        ],
        "usage": {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
        },
        "system_fingerprint": None,  # Optional: backend configuration fingerprint
    }


def build_openai_stream_chunk(
    chunk_dict: dict[str, Any],
    ctx: RequestContext,
    *,
    created_ts: int,
    include_role: bool,
) -> dict[str, Any]:
    """Convert a native streaming chunk into an OpenAI chat completion chunk.

    Builds a fully OpenAI-compliant streaming chunk including all required
    fields per the OpenAI API specification.

    Args:
        chunk_dict: Chunk dictionary from streaming use case.
        ctx: Request context with request_id.
        created_ts: Unix timestamp for the response.
        include_role: Whether to include role in the first delta.

    Returns:
        OpenAI-compliant chat completion chunk dictionary.
    """
    chunk_text = chunk_dict.get("chunk") or ""
    model_used = chunk_dict.get("model", "unknown")
    done = bool(chunk_dict.get("done"))
    finish_reason = chunk_dict.get("finish_reason", "stop" if done else None)

    delta: dict[str, Any] = {}
    if include_role and chunk_dict.get("role"):
        delta["role"] = chunk_dict["role"]
    if chunk_text:
        delta["content"] = chunk_text

    choice: dict[str, Any] = {
        "index": 0,
        "delta": delta,
        "logprobs": None,  # Required: null unless logprobs requested
        "finish_reason": finish_reason if done else None,
    }

    payload: dict[str, Any] = {
        "id": ctx.request_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model_used,
        "choices": [choice],
        "system_fingerprint": None,  # Optional: backend configuration fingerprint
    }

    if done:
        prompt_tokens = chunk_dict.get("prompt_eval_count")
        completion_tokens = chunk_dict.get("generation_eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            usage = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            }
            payload["usage"] = usage

    return payload
