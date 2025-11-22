"""Request and response models for the REST API.

This module defines Pydantic v2 models for all REST API request and response
schemas. All models use Pydantic's validation and serialization features for
type safety, input validation, and automatic API documentation generation.

Design Principles:
    - Type Safety: All models use Pydantic v2 with strict type validation
    - Input Validation: Request models validate and reject invalid input
    - API Documentation: Models auto-generate OpenAPI/Swagger documentation
    - Compatibility: Supports both native Ollama and OpenAI-compatible formats
    - Tool Calling: Full POML and OpenAI function calling support

Key Behaviors:
    - All models use Pydantic v2 with ConfigDict for configuration
    - Request models validate input and reject extra fields (extra="forbid")
    - Response models include comprehensive metadata (latency, metrics, etc.)
    - Streaming models allow extra fields for flexibility
    - All numeric fields have appropriate constraints (ge, le)
    - String fields are automatically stripped of whitespace

Validation Rules:
    - String fields: Automatically stripped of whitespace
    - Numeric fields: Range constraints where applicable (temperature, dimensions)
    - Required fields: Marked with Field(...)
    - Optional fields: Use Field(None, ...) with appropriate defaults
    - Image formats: Validated data URL format (data:image/...;base64,...)

Key Models:
    - Request Models: GenerateRequest, ChatRequest, VLMRequest, VLMRequestOpenAI
    - Response Models: GenerateResponse, ChatResponse, VLMResponse
    - Tool Models: Tool, ToolFunction, ToolCall (POML compatible)
    - Format Models: ResponseFormat (OpenAI compatible)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from shared_ollama.core.utils import get_default_text_model, get_default_vlm_model

# ============================================================================
# Tool/Function Calling Models (for POML and OpenAI compatibility)
# ============================================================================


class ToolFunction(BaseModel):
    """Function definition for tool calling.

    Defines a function that the model can call, including its schema.
    Compatible with both POML <tool-definition> and OpenAI function calling.

    Attributes:
        name: Function name. Must be valid identifier.
        description: Function description to help model understand when to call it.
        parameters: JSON schema defining function parameters (JSON Schema format).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    name: str = Field(..., min_length=1, description="Function name")
    description: str | None = Field(None, description="Function description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for function parameters"
    )


class Tool(BaseModel):
    """Tool definition for function calling.

    Wrapper around ToolFunction following OpenAI's tool format.
    Used in chat/VLM requests to define available tools.

    Attributes:
        type: Tool type. Currently only "function" is supported.
        function: Function definition (ToolFunction).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["function"] = Field(
        default="function", description="Tool type (only 'function' supported)"
    )
    function: ToolFunction = Field(..., description="Function definition")


class ToolCallFunction(BaseModel):
    """Function call details in a tool call.

    Contains the function name and arguments for a tool call made by the model.

    Attributes:
        name: Name of the function being called.
        arguments: JSON string containing function arguments.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    name: str = Field(..., min_length=1, description="Function name")
    arguments: str = Field(..., description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Tool call made by the model.

    Represents a function call that the model wants to make.
    Corresponds to POML's <tool-request> element.

    Attributes:
        id: Unique identifier for this tool call.
        type: Tool call type. Currently only "function" is supported.
        function: Function call details (ToolCallFunction).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    id: str = Field(..., min_length=1, description="Unique tool call ID")
    type: Literal["function"] = Field(default="function", description="Tool call type")
    function: ToolCallFunction = Field(..., description="Function call details")


# ============================================================================
# Request/Response Models
# ============================================================================


class ResponseFormat(BaseModel):
    """OpenAI-compatible response_format wrapper."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["text", "json_object", "json_schema"] = Field(
        ...,
        description="Response format type ('text', 'json_object', or 'json_schema')",
    )
    json_schema: dict[str, Any] | None = Field(
        None,
        description=(
            "JSON schema definition required when type='json_schema'. "
            'Supports both OpenAI\'s {"name": ..., "schema": {...}} structure '
            "and direct JSON Schema objects."
        ),
    )

    @field_validator("json_schema")
    @classmethod
    def validate_schema_for_type(
        cls,
        value: dict[str, Any] | None,
        info: ValidationInfo,
    ) -> dict[str, Any] | None:
        """Ensure json_schema is provided when required."""
        if info.data.get("type") == "json_schema" and not value:
            raise ValueError("json_schema is required when type='json_schema'")
        return value


class GenerateRequest(BaseModel):
    """Request model for text generation endpoint.

    Validates and deserializes POST /api/v1/generate request bodies.
    All optional generation parameters default to None (model defaults used).

    Attributes:
        prompt: Text prompt for generation. Required, must not be empty.
        model: Model name. Optional, defaults to service default.
        system: System message to set model behavior. Optional.
        stream: Whether to stream the response. Defaults to False.
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        response_format: OpenAI-compatible response format. Optional.
        tools: Tools/functions the model can call (POML compatible). Optional.
        temperature: Sampling temperature (0.0-2.0). Optional.
        top_p: Nucleus sampling parameter (0.0-1.0). Optional.
        top_k: Top-k sampling parameter (>=1). Optional.
        max_tokens: Maximum tokens to generate (>=1). Optional.
        seed: Random seed for reproducibility. Optional.
        stop: List of stop sequences. Optional.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    prompt: str = Field(..., min_length=1, description="The prompt to generate text from")
    model: str | None = Field(None, description="Model to use (defaults to service default)")
    system: str | None = Field(None, description="System message for the model")
    stream: bool = Field(False, description="Whether to stream the response")
    format: str | dict[str, Any] | None = Field(
        None,
        description="(Deprecated) Native Ollama format field. Prefer response_format for OpenAI compatibility.",
    )
    response_format: ResponseFormat | None = Field(
        None,
        description="OpenAI-compatible response_format. Overrides format when provided.",
    )
    tools: list[Tool] | None = Field(
        None, description="Tools/functions the model can call (POML compatible)"
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")


class ChatMessage(BaseModel):
    """A chat message with role and text content (native Ollama format).

    Simple text-only message for chat completion. For vision language models
    with images, use the dedicated /api/v1/vlm endpoint with VLMRequest.

    Supports tool calls for function calling workflows (POML compatible).

    Attributes:
        role: Message role. Must be "user", "assistant", "system", or "tool".
        content: Text content of the message. Optional when tool_calls present.
        tool_calls: List of tool calls made by the assistant. Optional.
        tool_call_id: ID of the tool call this message is responding to (for role="tool"). Optional.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    role: Literal["user", "assistant", "system", "tool"] = Field(
        ..., description="Message role: 'user', 'assistant', 'system', or 'tool'"
    )
    content: str | None = Field(
        None, description="Text content of the message (optional if tool_calls present)"
    )
    tool_calls: list[ToolCall] | None = Field(None, description="Tool calls made by the assistant")
    tool_call_id: str | None = Field(
        None, description="Tool call ID this message responds to (for role='tool')"
    )

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_or_tool_calls(cls, v: str | None, info: Any) -> str | None:
        """Validate that either content or tool_calls is present."""
        tool_calls = info.data.get("tool_calls")
        if v is None and not tool_calls:
            raise ValueError("Message must have either content or tool_calls")
        return v


class ChatRequest(BaseModel):
    """Request model for chat completion endpoint.

    Validates and deserializes POST /api/v1/chat request bodies.

    Attributes:
        messages: List of chat messages. Required, must contain at least one.
        model: Model name. Optional, defaults to service default.
        stream: Whether to stream the response. Defaults to False.
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        tools: List of tools/functions the model can call. Optional (POML compatible).
        temperature: Sampling temperature (0.0-2.0). Optional.
        top_p: Nucleus sampling parameter (0.0-1.0). Optional.
        top_k: Top-k sampling parameter (>=1). Optional.
        max_tokens: Maximum tokens to generate (>=1). Optional.
        seed: Random seed for reproducibility. Optional.
        stop: List of stop sequences. Optional.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    messages: list[ChatMessage] = Field(..., min_length=1, description="List of chat messages")
    model: str | None = Field(None, description="Model to use (defaults to service default)")
    stream: bool = Field(False, description="Whether to stream the response")
    format: str | dict[str, Any] | None = Field(
        None,
        description="(Deprecated) Native Ollama format field. Prefer response_format for OpenAI compatibility.",
    )
    response_format: ResponseFormat | None = Field(
        None,
        description="OpenAI-compatible response_format. Overrides format when provided.",
    )
    tools: list[Tool] | None = Field(
        None, description="Tools/functions the model can call (POML compatible)"
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")


class GenerateResponse(BaseModel):
    """Response model for text generation endpoint.

    Contains generated text and comprehensive performance metrics.
    All time values are in milliseconds.

    Attributes:
        text: Generated text content.
        model: Model name used for generation.
        request_id: Unique request identifier for tracking.
        latency_ms: Total request latency (>=0.0).
        model_load_ms: Model load time in milliseconds (>=0.0). None if N/A.
        model_warm_start: Whether model was already loaded (no load time).
        prompt_eval_count: Number of prompt tokens evaluated (>=0). None if N/A.
        generation_eval_count: Number of generation tokens produced (>=0). None if N/A.
        total_duration_ms: Total generation duration in milliseconds (>=0.0). None if N/A.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used for generation")
    request_id: str = Field(..., description="Unique request identifier")
    latency_ms: float = Field(..., ge=0.0, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, ge=0.0, description="Model load time in milliseconds")
    model_warm_start: bool = Field(..., description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(
        None, ge=0, description="Number of prompt tokens evaluated"
    )
    generation_eval_count: int | None = Field(
        None, ge=0, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, ge=0.0, description="Total generation duration in milliseconds"
    )


class ChatResponse(BaseModel):
    """Response model for chat completion endpoint.

    Contains assistant's response message and performance metrics.
    All time values are in milliseconds.

    Attributes:
        message: Assistant's response message (ChatMessage).
        model: Model name used for generation.
        request_id: Unique request identifier for tracking.
        latency_ms: Total request latency (>=0.0).
        model_load_ms: Model load time in milliseconds (>=0.0). None if N/A.
        model_warm_start: Whether model was already loaded (no load time).
        prompt_eval_count: Number of prompt tokens evaluated (>=0). None if N/A.
        generation_eval_count: Number of generation tokens produced (>=0). None if N/A.
        total_duration_ms: Total generation duration in milliseconds (>=0.0). None if N/A.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    message: ChatMessage = Field(..., description="Assistant's response message")
    model: str = Field(..., description="Model used for generation")
    request_id: str = Field(..., description="Unique request identifier")
    latency_ms: float = Field(..., ge=0.0, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, ge=0.0, description="Model load time in milliseconds")
    model_warm_start: bool = Field(..., description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(
        None, ge=0, description="Number of prompt tokens evaluated"
    )
    generation_eval_count: int | None = Field(
        None, ge=0, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, ge=0.0, description="Total generation duration in milliseconds"
    )


class GenerateStreamChunk(BaseModel):
    """Streaming chunk for generate endpoint.

    Represents a single chunk in a streaming generation response.
    Metrics fields are only present in the final chunk (done=True).

    Attributes:
        chunk: Incremental text chunk. Empty string in final chunk if no new text.
        done: Whether generation is complete. True for final chunk.
        model: Model name. Present in all chunks.
        request_id: Request identifier. Present in all chunks.
        latency_ms: Total request latency (>=0.0). Only in final chunk.
        model_load_ms: Model load time in milliseconds (>=0.0). Only in final chunk.
        model_warm_start: Whether model was already loaded. Only in final chunk.
        prompt_eval_count: Number of prompt tokens evaluated (>=0). Only in final chunk.
        generation_eval_count: Number of generation tokens produced (>=0). Only in final chunk.
        total_duration_ms: Total generation duration in milliseconds (>=0.0). Only in final chunk.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for streaming flexibility
    )

    chunk: str = Field(..., description="Incremental text chunk")
    done: bool = Field(..., description="Whether generation is complete")
    model: str | None = Field(None, description="Model name")
    request_id: str | None = Field(None, description="Request ID")
    latency_ms: float | None = Field(None, ge=0.0, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, ge=0.0, description="Model load time in milliseconds")
    model_warm_start: bool | None = Field(None, description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(
        None, ge=0, description="Number of prompt tokens evaluated"
    )
    generation_eval_count: int | None = Field(
        None, ge=0, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, ge=0.0, description="Total generation duration in milliseconds"
    )


class ChatStreamChunk(BaseModel):
    """Streaming chunk for chat endpoint.

    Represents a single chunk in a streaming chat completion response.
    Metrics fields are only present in the final chunk (done=True).

    Attributes:
        chunk: Incremental message content. Empty string in final chunk if no new text.
        role: Message role. Defaults to "assistant".
        done: Whether response is complete. True for final chunk.
        model: Model name. Present in all chunks.
        request_id: Request identifier. Present in all chunks.
        latency_ms: Total request latency (>=0.0). Only in final chunk.
        model_load_ms: Model load time in milliseconds (>=0.0). Only in final chunk.
        model_warm_start: Whether model was already loaded. Only in final chunk.
        prompt_eval_count: Number of prompt tokens evaluated (>=0). Only in final chunk.
        generation_eval_count: Number of generation tokens produced (>=0). Only in final chunk.
        total_duration_ms: Total generation duration in milliseconds (>=0.0). Only in final chunk.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for streaming flexibility
    )

    chunk: str = Field(..., description="Incremental message content")
    role: Literal["user", "assistant", "system"] = Field(
        default="assistant", description="Message role"
    )
    done: bool = Field(..., description="Whether response is complete")
    model: str | None = Field(None, description="Model name")
    request_id: str | None = Field(None, description="Request ID")
    latency_ms: float | None = Field(None, ge=0.0, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, ge=0.0, description="Model load time in milliseconds")
    model_warm_start: bool | None = Field(None, description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(
        None, ge=0, description="Number of prompt tokens evaluated"
    )
    generation_eval_count: int | None = Field(
        None, ge=0, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, ge=0.0, description="Total generation duration in milliseconds"
    )


class ModelInfo(BaseModel):
    """Information about an available model.

    Represents metadata for a single model available in the Ollama service.

    Attributes:
        name: Model name (e.g., "qwen3-vl:8b-instruct-q4_K_M").
        size: Model size in bytes (>=0). None if size information unavailable.
        modified_at: Last modification time as ISO 8601 string. None if unavailable.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    name: str = Field(..., description="Model name")
    size: int | None = Field(None, ge=0, description="Model size in bytes")
    modified_at: str | None = Field(None, description="Last modification time")


class ModelsResponse(BaseModel):
    """Response model for listing models endpoint.

    Contains the list of all available models in the Ollama service.

    Attributes:
        models: List of ModelInfo objects, one for each available model.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    models: list[ModelInfo] = Field(..., description="List of available models")


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Contains service health status and version information.

    Attributes:
        status: Service status. Must be "healthy" or "unhealthy".
        ollama_service: Ollama service status string. May include error details.
        version: API version string (e.g., "1.0.0").
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    status: Literal["healthy", "unhealthy"] = Field(
        ..., description="Service status: 'healthy' or 'unhealthy'"
    )
    ollama_service: str = Field(..., description="Ollama service status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for error responses.

    Standardized error response format returned by error handlers.

    Attributes:
        error: Human-readable error message.
        error_type: Type/category of error (e.g., "ValidationError", "HTTPError").
            None if error type not available.
        request_id: Request identifier for tracking. None if request ID not available.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    error: str = Field(..., description="Error message")
    error_type: str | None = Field(None, description="Error type")
    request_id: str | None = Field(None, description="Request identifier if available")


class QueueStatsResponse(BaseModel):
    """Response model for queue statistics endpoint.

    Contains comprehensive queue metrics and configuration.

    Attributes:
        queued: Number of requests currently waiting in queue (>=0).
        in_progress: Number of requests currently being processed (>=0).
        completed: Total requests completed since startup (>=0).
        failed: Total requests failed since startup (>=0).
        rejected: Total requests rejected due to full queue (>=0).
        timeout: Total requests timed out waiting in queue (>=0).
        total_wait_time_ms: Cumulative time all requests spent waiting (>=0.0).
        max_wait_time_ms: Maximum wait time observed for any request (>=0.0).
        avg_wait_time_ms: Average wait time per request (>=0.0).
        max_concurrent: Maximum concurrent requests allowed (>=1).
        max_queue_size: Maximum queue size (>=1).
        default_timeout: Default queue timeout in seconds (>=0.0).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    queued: int = Field(..., ge=0, description="Number of requests currently waiting in queue")
    in_progress: int = Field(..., ge=0, description="Number of requests currently being processed")
    completed: int = Field(..., ge=0, description="Total requests completed since startup")
    failed: int = Field(..., ge=0, description="Total requests failed since startup")
    rejected: int = Field(
        ..., ge=0, description="Total requests rejected (queue full) since startup"
    )
    timeout: int = Field(
        ..., ge=0, description="Total requests timed out waiting in queue since startup"
    )
    total_wait_time_ms: float = Field(
        ..., ge=0.0, description="Total time all requests spent waiting (ms)"
    )
    max_wait_time_ms: float = Field(..., ge=0.0, description="Maximum wait time observed (ms)")
    avg_wait_time_ms: float = Field(..., ge=0.0, description="Average wait time per request (ms)")
    max_concurrent: int = Field(..., ge=1, description="Maximum concurrent requests allowed")
    max_queue_size: int = Field(..., ge=1, description="Maximum queue size")
    default_timeout: float = Field(..., ge=0.0, description="Default queue timeout (seconds)")


class VLMMessage(BaseModel):
    """Simple text-only message for VLM requests (native Ollama format).

    VLM requests use native Ollama format with separate images parameter.
    Messages contain only text content - images are passed separately.

    Supports tool calls for function calling workflows (POML compatible).

    Attributes:
        role: Message role. Must be "user", "assistant", "system", or "tool".
        content: Text content of the message. Optional when tool_calls present.
        tool_calls: List of tool calls made by the assistant. Optional.
        tool_call_id: ID of the tool call this message is responding to (for role="tool"). Optional.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    role: Literal["user", "assistant", "system", "tool"] = Field(
        ..., description="Message role: 'user', 'assistant', 'system', or 'tool'"
    )
    content: str | None = Field(
        None, description="Text content of the message (optional if tool_calls present)"
    )
    tool_calls: list[ToolCall] | None = Field(None, description="Tool calls made by the assistant")
    tool_call_id: str | None = Field(
        None, description="Tool call ID this message responds to (for role='tool')"
    )

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_or_tool_calls(cls, v: str | None, info: Any) -> str | None:
        """Validate that either content or tool_calls is present."""
        tool_calls = info.data.get("tool_calls")
        if v is None and not tool_calls:
            raise ValueError("Message must have either content or tool_calls")
        return v


class VLMRequest(BaseModel):
    """Request model for VLM (Vision-Language Model) endpoint.

    Uses native Ollama format with separate images parameter.
    Vision-language model endpoint that requires at least one image.
    For text-only requests, use /api/v1/chat instead.

    Attributes:
        messages: List of text-only chat messages. Required.
        images: List of base64-encoded images (data URLs). Required, min 1 image.
        model: VLM model name (defaults to configured profile based on system hardware).
        stream: Whether to stream the response. Defaults to False.
        format: Output format specification. Can be:
            - "json" for JSON mode
            - dict with JSON schema for structured output
            - None for default text output
        tools: List of tools/functions the model can call. Optional (POML compatible).
        temperature: Sampling temperature (0.0-2.0). Optional.
        top_p: Nucleus sampling parameter (0.0-1.0). Optional.
        top_k: Top-k sampling parameter (>=1). Optional.
        max_tokens: Maximum tokens to generate (>=1). Optional.
        seed: Random seed for reproducibility. Optional.
        stop: List of stop sequences. Optional.
        image_compression: Enable image compression (recommended). Defaults to True.
        max_dimension: Maximum image dimension for resizing (256-2667). Defaults to 2667.
        compression_format: Image compression format (jpeg, png, or webp). Defaults to jpeg.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    messages: list[VLMMessage] = Field(
        ..., min_length=1, description="List of text-only chat messages (native Ollama format)"
    )
    images: list[str] = Field(
        ...,
        min_length=1,
        description="List of base64-encoded images as data URLs (data:image/...;base64,...)",
    )
    model: str | None = Field(
        default_factory=get_default_vlm_model,
        description="VLM model (defaults to configured profile based on system hardware)",
    )
    stream: bool = Field(False, description="Whether to stream the response")
    format: str | dict[str, Any] | None = Field(
        None,
        description="(Deprecated) Native Ollama format field. Prefer response_format for OpenAI compatibility.",
    )
    response_format: ResponseFormat | None = Field(
        None,
        description="OpenAI-compatible response_format. Overrides format when provided.",
    )
    tools: list[Tool] | None = Field(
        None, description="Tools/functions the model can call (POML compatible)"
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")
    image_compression: bool = Field(True, description="Enable image compression (recommended)")
    max_dimension: int = Field(
        2667, ge=256, le=2667, description="Maximum image dimension for resizing"
    )
    compression_format: Literal["jpeg", "png", "webp"] = Field(
        "jpeg", description="Image compression format (jpeg, png, or webp)"
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: list[str]) -> list[str]:
        """Validate image data URLs."""
        for idx, img in enumerate(v):
            if not img.startswith("data:image/"):
                raise ValueError(f"Image {idx}: must start with 'data:image/'")
            if ";base64," not in img:
                raise ValueError(f"Image {idx}: must contain ';base64,' separator")
        return v


# ============================================================================
# OpenAI-Compatible VLM Models (for Docling and other OpenAI-compatible clients)
# ============================================================================


class ImageURL(BaseModel):
    """Image URL wrapper for OpenAI-compatible format.

    Attributes:
        url: Base64-encoded image data URL (data:image/...;base64,...).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    url: str = Field(..., min_length=1, description="Base64-encoded image data URL")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate image data URL format."""
        if not v.startswith("data:image/"):
            raise ValueError("Image URL must start with 'data:image/'")
        if ";base64," not in v:
            raise ValueError("Image URL must contain ';base64,' separator")
        return v


class ImageContentPart(BaseModel):
    """Image content part for OpenAI-compatible multimodal messages.

    Attributes:
        type: Content type, must be "image_url".
        image_url: Image URL object containing base64-encoded image.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["image_url"] = Field(..., description="Content type: 'image_url'")
    image_url: ImageURL = Field(..., description="Image URL object")


class TextContentPart(BaseModel):
    """Text content part for OpenAI-compatible multimodal messages.

    Attributes:
        type: Content type, must be "text".
        text: Text content string.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["text"] = Field(..., description="Content type: 'text'")
    text: str = Field(..., min_length=1, description="Text content")


# Union type for content parts
ContentPart = ImageContentPart | TextContentPart


class ChatMessageOpenAI(BaseModel):
    """OpenAI-compatible chat message with multimodal content support.

    Supports both simple string content and multimodal content (text + images).
    Used for OpenAI-compatible VLM and chat endpoints. Supports tool calling
    for function calling workflows.

    Attributes:
        role: Message role. Must be "user", "assistant", "system", "developer", or "tool".
            The "developer" role is OpenAI's newer equivalent of "system" for setting
            model behavior instructions.
        content: Either a string (text-only) or list of content parts (multimodal).
            Optional when tool_calls present (for assistant messages).
        tool_calls: List of tool calls made by the assistant. Optional.
        tool_call_id: ID of the tool call this message is responding to (for role="tool"). Optional.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ..., description="Message role: 'user', 'assistant', 'system', 'developer', or 'tool'"
    )
    content: str | list[ContentPart] | None = Field(
        None,
        description="Message content: string (text-only) or list of content parts (multimodal). Optional if tool_calls present.",
    )
    tool_calls: list[ToolCall] | None = Field(
        None, description="Tool calls made by the assistant (for role='assistant')"
    )
    tool_call_id: str | None = Field(
        None, description="Tool call ID this message responds to (for role='tool')"
    )

    @field_validator("content", mode="after")
    @classmethod
    def validate_content_or_tool_calls(
        cls, v: str | list[ContentPart] | None, info: ValidationInfo
    ) -> str | list[ContentPart] | None:
        """Validate that either content or tool_calls is present."""
        tool_calls = info.data.get("tool_calls")
        if v is None and not tool_calls:
            raise ValueError("Message must have either content or tool_calls")
        if isinstance(v, str) and not v.strip() and not tool_calls:
            raise ValueError("Text content cannot be empty")
        if isinstance(v, list) and not v:
            raise ValueError("Content parts list cannot be empty")
        return v


class ChatRequestOpenAI(BaseModel):
    """OpenAI-compatible text-only chat request model.

    Uses OpenAI-compatible message format for text-only chat completions.
    Converted internally to native Ollama format for processing.

    For OpenAI-compatible clients that expect /chat/completions endpoint.

    Attributes:
        messages: List of OpenAI-compatible chat messages (text-only).
        model: Model name (defaults to configured profile based on system hardware).
        stream: Whether to stream the response. Defaults to False.
        temperature: Sampling temperature (0.0-2.0). Optional.
        top_p: Nucleus sampling parameter (0.0-1.0). Optional.
        top_k: Top-k sampling parameter (>=1). Optional.
        max_tokens: Maximum tokens to generate (>=1). Optional.
        seed: Random seed for reproducibility. Optional.
        stop: List of stop sequences. Optional.
        tools: Tools/functions the model can call (POML compatible). Optional.
        response_format: OpenAI-compatible response format. Optional.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    messages: list[ChatMessageOpenAI] = Field(
        ..., min_length=1, description="List of OpenAI-compatible chat messages"
    )
    model: str | None = Field(
        default_factory=get_default_text_model,
        description="Model name (defaults to configured profile based on system hardware)",
    )
    stream: bool = Field(False, description="Whether to stream the response")
    format: str | dict[str, Any] | None = Field(
        None,
        description="(Deprecated) Native Ollama format field. Prefer response_format for OpenAI compatibility.",
    )
    response_format: ResponseFormat | None = Field(
        None,
        description="OpenAI-compatible response_format. Overrides format when provided.",
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="(Deprecated) Maximum tokens to generate. Prefer max_completion_tokens.",
    )
    max_completion_tokens: int | None = Field(
        None,
        ge=1,
        description="Maximum tokens to generate (OpenAI's preferred parameter). Overrides max_tokens when provided.",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")
    tools: list[Tool] | None = Field(
        None, description="Tools/functions the model can call (POML compatible)"
    )

    @property
    def effective_max_tokens(self) -> int | None:
        """Get the effective max tokens value, preferring max_completion_tokens."""
        return (
            self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens
        )

    @field_validator("messages")
    @classmethod
    def validate_text_only(cls, v: list[ChatMessageOpenAI]) -> list[ChatMessageOpenAI]:
        """Validate that messages are text-only (no images)."""
        for msg in v:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, ImageContentPart):
                        raise ValueError(
                            "ChatRequestOpenAI does not support images. "
                            "For multimodal requests with images, use /api/v1/vlm/openai endpoint instead."
                        )
        return v


class VLMRequestOpenAI(BaseModel):
    """OpenAI-compatible VLM request model.

    Uses OpenAI-compatible multimodal message format where images are embedded
    in message content as image_url parts. Converted internally to native Ollama
    format (separate images parameter) for processing.

    For Docling and other OpenAI-compatible clients.

    Attributes:
        messages: List of OpenAI-compatible chat messages with multimodal content.
        model: VLM model name (defaults to configured profile based on system hardware).
        stream: Whether to stream the response. Defaults to False.
        temperature: Sampling temperature (0.0-2.0). Optional.
        top_p: Nucleus sampling parameter (0.0-1.0). Optional.
        top_k: Top-k sampling parameter (>=1). Optional.
        max_tokens: Maximum tokens to generate (>=1). Optional.
        seed: Random seed for reproducibility. Optional.
        stop: List of stop sequences. Optional.
        tools: Tools/functions the model can call (POML compatible). Optional.
        image_compression: Enable image compression (recommended). Defaults to True.
        max_dimension: Maximum image dimension for resizing (256-2667). Defaults to 2667.
        compression_format: Image compression format (jpeg, png, or webp). Defaults to jpeg.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    messages: list[ChatMessageOpenAI] = Field(
        ..., min_length=1, description="List of OpenAI-compatible chat messages"
    )
    model: str | None = Field(
        default_factory=get_default_vlm_model,
        description="VLM model (defaults to configured profile based on system hardware)",
    )
    stream: bool = Field(False, description="Whether to stream the response")
    format: str | dict[str, Any] | None = Field(
        None,
        description="(Deprecated) Native Ollama format field. Prefer response_format for OpenAI compatibility.",
    )
    response_format: ResponseFormat | None = Field(
        None,
        description="OpenAI-compatible response_format. Overrides format when provided.",
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="(Deprecated) Maximum tokens to generate. Prefer max_completion_tokens.",
    )
    max_completion_tokens: int | None = Field(
        None,
        ge=1,
        description="Maximum tokens to generate (OpenAI's preferred parameter). Overrides max_tokens when provided.",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")
    tools: list[Tool] | None = Field(
        None, description="Tools/functions the model can call (POML compatible)"
    )
    image_compression: bool = Field(True, description="Enable image compression (recommended)")
    max_dimension: int = Field(
        2667, ge=256, le=2667, description="Maximum image dimension for resizing"
    )
    compression_format: Literal["jpeg", "png", "webp"] = Field(
        "jpeg", description="Image compression format (jpeg, png, or webp)"
    )

    @property
    def effective_max_tokens(self) -> int | None:
        """Get the effective max tokens value, preferring max_completion_tokens."""
        return (
            self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens
        )

    @field_validator("messages")
    @classmethod
    def validate_has_images(cls, v: list[ChatMessageOpenAI]) -> list[ChatMessageOpenAI]:
        """Validate that at least one message contains an image."""
        has_image = False
        for msg in v:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, ImageContentPart):
                        has_image = True
                        break
            if has_image:
                break

        if not has_image:
            raise ValueError(
                "VLM requests must contain at least one image. "
                "For text-only requests, use /api/v1/chat/completions endpoint instead."
            )
        return v


class VLMResponse(BaseModel):
    """Response model for VLM endpoint.

    Contains assistant's response message and VLM-specific performance metrics.
    All time values are in milliseconds.

    Attributes:
        message: Assistant's response message.
        model: Model used for generation.
        request_id: Unique request identifier.
        latency_ms: Request latency in milliseconds.
        model_load_ms: Model load time in milliseconds. None if N/A.
        model_warm_start: Whether model was already loaded.
        images_processed: Number of images processed.
        compression_savings_bytes: Bytes saved by compression. None if compression disabled.
        prompt_eval_count: Number of prompt tokens evaluated. None if N/A.
        generation_eval_count: Number of generation tokens produced. None if N/A.
        total_duration_ms: Total generation duration in milliseconds. None if N/A.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    message: ChatMessage = Field(..., description="Assistant's response message")
    model: str = Field(..., description="Model used for generation")
    request_id: str = Field(..., description="Unique request identifier")
    latency_ms: float = Field(..., ge=0.0, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, ge=0.0, description="Model load time in milliseconds")
    model_warm_start: bool = Field(..., description="Whether model was already loaded")
    images_processed: int = Field(..., ge=1, description="Number of images processed")
    compression_savings_bytes: int | None = Field(None, description="Bytes saved by compression")
    prompt_eval_count: int | None = Field(
        None, ge=0, description="Number of prompt tokens evaluated"
    )
    generation_eval_count: int | None = Field(
        None, ge=0, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, ge=0.0, description="Total generation duration in milliseconds"
    )


class BatchChatRequest(BaseModel):
    """Request model for batch text-only chat processing.

    Process multiple text-only chat requests concurrently.
    Maximum 50 requests per batch for fairness.

    Attributes:
        requests: List of chat requests to process (1-50).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    requests: list[ChatRequest] = Field(
        ..., min_length=1, max_length=50, description="List of chat requests (max 50)"
    )


class BatchChatRequestOpenAI(BaseModel):
    """OpenAI-compatible batch text-only chat processing request.

    Process multiple OpenAI-compatible chat requests concurrently.
    Maximum 50 requests per batch for fairness.

    Attributes:
        requests: List of OpenAI-compatible chat requests to process (1-50).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    requests: list[ChatRequestOpenAI] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of OpenAI-compatible chat requests (max 50)",
    )


class BatchVLMRequestOpenAI(BaseModel):
    """OpenAI-compatible batch VLM processing request.

    Process multiple OpenAI-compatible VLM requests concurrently.
    Maximum 20 requests per batch due to image processing overhead.

    Attributes:
        requests: List of OpenAI-compatible VLM requests to process (1-20).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    requests: list[VLMRequestOpenAI] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of OpenAI-compatible VLM requests (max 20)",
    )


class BatchVLMRequest(BaseModel):
    """Request model for batch VLM processing.

    Process multiple VLM requests concurrently.
    Maximum 20 requests per batch due to image processing overhead.

    Attributes:
        requests: List of VLM requests to process (1-20).
        compression_format: Image compression format for all requests.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    requests: list[VLMRequest] = Field(
        ..., min_length=1, max_length=20, description="List of VLM requests (max 20)"
    )
    compression_format: Literal["jpeg", "png", "webp"] = Field(
        "jpeg", description="Image compression format for all requests (jpeg, png, or webp)"
    )


class BatchResponse(BaseModel):
    """Response model for batch processing endpoints.

    Contains results for all requests in the batch with aggregate statistics.

    Attributes:
        batch_id: Unique batch identifier.
        total_requests: Total requests in batch.
        successful: Successfully processed requests.
        failed: Failed requests.
        total_time_ms: Total batch processing time in milliseconds.
        results: Individual request results (index, success/failure, data/error).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    batch_id: str = Field(..., description="Unique batch identifier")
    total_requests: int = Field(..., ge=1, description="Total requests in batch")
    successful: int = Field(..., ge=0, description="Successfully processed requests")
    failed: int = Field(..., ge=0, description="Failed requests")
    total_time_ms: float = Field(..., ge=0.0, description="Total batch processing time")
    results: list[dict[str, Any]] = Field(..., description="Individual request results")


@dataclass(slots=True, frozen=True)
class RequestContext:
    """Context for tracking API requests.

    Immutable context object extracted from FastAPI request. Used for
    logging, metrics, and request tracking throughout request lifecycle.

    Attributes:
        request_id: Unique request identifier (UUID string).
        client_ip: Client IP address extracted from request.
        user_agent: User-Agent header value. None if not present.
        project_name: Project name from X-Project-Name header. None if not present.
    """

    request_id: str
    client_ip: str
    user_agent: str | None = None
    project_name: str | None = None


# ============================================================================
# Observability / Telemetry Response Models
# ============================================================================


class MetricsResponse(BaseModel):
    """Structured response for /metrics endpoint."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    total_requests: int = Field(..., ge=0, description="Total requests observed")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    requests_by_model: dict[str, int] = Field(
        default_factory=dict, description="Request counts grouped by model"
    )
    requests_by_operation: dict[str, int] = Field(
        default_factory=dict, description="Request counts grouped by operation"
    )
    average_latency_ms: float = Field(..., ge=0.0, description="Average latency (ms)")
    p50_latency_ms: float = Field(..., ge=0.0, description="50th percentile latency (ms)")
    p95_latency_ms: float = Field(..., ge=0.0, description="95th percentile latency (ms)")
    p99_latency_ms: float = Field(..., ge=0.0, description="99th percentile latency (ms)")
    errors_by_type: dict[str, int] = Field(
        default_factory=dict, description="Error counts grouped by error type"
    )
    last_request_time: datetime | None = Field(None, description="Timestamp of most recent request")
    first_request_time: datetime | None = Field(
        None, description="Timestamp of earliest request included"
    )


class PerformanceModelStats(BaseModel):
    """Per-model performance statistics."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    avg_tokens_per_second: float = Field(..., ge=0.0, description="Average tokens/sec")
    avg_load_time_ms: float = Field(..., ge=0.0, description="Average load time (ms)")
    avg_generation_time_ms: float = Field(..., ge=0.0, description="Average generation time (ms)")
    request_count: int = Field(..., ge=0, description="Number of successful requests")


class PerformanceStatsResponse(BaseModel):
    """Structured response for /performance/stats endpoint."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    avg_tokens_per_second: float = Field(..., ge=0.0, description="Overall avg tokens/sec")
    avg_load_time_ms: float = Field(..., ge=0.0, description="Overall avg load time (ms)")
    avg_generation_time_ms: float = Field(
        ..., ge=0.0, description="Overall avg generation time (ms)"
    )
    total_requests: int = Field(..., ge=0, description="Successful requests counted")
    by_model: dict[str, PerformanceModelStats] = Field(
        default_factory=dict, description="Per-model performance statistics"
    )


class ProjectMetricsResponse(BaseModel):
    """Per-project analytics details."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    project_name: str = Field(..., description="Project identifier")
    total_requests: int = Field(..., ge=0, description="Total requests for project")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    requests_by_model: dict[str, int] = Field(
        default_factory=dict, description="Model usage for project"
    )
    requests_by_operation: dict[str, int] = Field(
        default_factory=dict, description="Operation breakdown for project"
    )
    average_latency_ms: float = Field(..., ge=0.0, description="Average latency (ms)")
    total_latency_ms: float = Field(..., ge=0.0, description="Total latency accumulated (ms)")
    last_request_time: datetime | None = Field(None, description="Most recent request timestamp")
    first_request_time: datetime | None = Field(None, description="Oldest request timestamp")


class HourlyMetricsEntry(BaseModel):
    """Hourly aggregated analytics entry."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    timestamp: datetime = Field(..., description="Hour bucket timestamp (UTC)")
    requests_count: int = Field(..., ge=0, description="Total requests in this hour")
    successful_count: int = Field(..., ge=0, description="Successful requests in this hour")
    failed_count: int = Field(..., ge=0, description="Failed requests in this hour")
    average_latency_ms: float = Field(..., ge=0.0, description="Average latency (ms)")
    requests_by_model: dict[str, int] = Field(
        default_factory=dict, description="Model usage during this hour"
    )


class AnalyticsResponse(BaseModel):
    """Structured response for /analytics endpoint."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    total_requests: int = Field(..., ge=0, description="Total requests observed")
    successful_requests: int = Field(..., ge=0, description="Successful requests")
    failed_requests: int = Field(..., ge=0, description="Failed requests")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    average_latency_ms: float = Field(..., ge=0.0, description="Average latency (ms)")
    p50_latency_ms: float = Field(..., ge=0.0, description="50th percentile latency")
    p95_latency_ms: float = Field(..., ge=0.0, description="95th percentile latency")
    p99_latency_ms: float = Field(..., ge=0.0, description="99th percentile latency")
    requests_by_model: dict[str, int] = Field(
        default_factory=dict, description="Requests grouped by model"
    )
    requests_by_operation: dict[str, int] = Field(
        default_factory=dict, description="Requests grouped by operation"
    )
    requests_by_project: dict[str, int] = Field(
        default_factory=dict, description="Requests grouped by project"
    )
    project_metrics: dict[str, ProjectMetricsResponse] = Field(
        default_factory=dict, description="Detailed metrics per project"
    )
    hourly_metrics: list[HourlyMetricsEntry] = Field(
        default_factory=list, description="Hourly aggregated metrics"
    )
    start_time: datetime | None = Field(None, description="Start of reporting window")
    end_time: datetime | None = Field(None, description="End of reporting window")
