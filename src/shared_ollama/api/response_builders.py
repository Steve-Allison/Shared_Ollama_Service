"""Shared response building utilities for route handlers.

This module provides helper functions to build consistent API responses,
reducing code duplication across route handlers.
"""

from __future__ import annotations

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
        message=ChatMessage(role="assistant", content=message_content),
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
        message=ChatMessage(role="assistant", content=message_content),
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

