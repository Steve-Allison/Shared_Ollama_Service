"""Request and response models for the REST API.

This module defines Pydantic v2 models for all REST API request and response
schemas. All models use Pydantic's validation and serialization features.

Key behaviors:
    - All models use Pydantic v2 with ConfigDict for configuration
    - Request models validate input and reject extra fields
    - Response models include comprehensive metadata (latency, metrics, etc.)
    - Streaming models allow extra fields for flexibility
    - All numeric fields have appropriate constraints (ge, le)

Validation:
    - String fields are automatically stripped of whitespace
    - Numeric fields have range constraints where applicable
    - Required fields are marked with Field(...)
    - Optional fields use Field(None, ...) with appropriate defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        description="Output format: 'json' for JSON mode, or JSON schema object for structured output",
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")


class ImageContentPart(BaseModel):
    """Image content part for multimodal messages.

    Represents an image in a multimodal message. Images must be base64-encoded
    with a data URL format: data:image/<format>;base64,<base64_data>

    Attributes:
        type: Content type, must be "image_url".
        image_url: Image URL object containing the base64-encoded image.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["image_url"] = Field(..., description="Content type: 'image_url'")
    image_url: dict[str, str] = Field(
        ..., description="Image URL object with 'url' key containing base64 data URL"
    )

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate image URL format."""
        if "url" not in v:
            raise ValueError("image_url must contain 'url' key")
        url = v["url"]
        if not url.startswith("data:image/"):
            raise ValueError("Image URL must start with 'data:image/'")
        if ";base64," not in url:
            raise ValueError("Image URL must contain ';base64,' separator")
        return v


class TextContentPart(BaseModel):
    """Text content part for multimodal messages.

    Represents text in a multimodal message.

    Attributes:
        type: Content type, must be "text".
        text: Text content.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    type: Literal["text"] = Field(..., description="Content type: 'text'")
    text: str = Field(..., min_length=1, description="Text content")


ContentPart = Union[TextContentPart, ImageContentPart]


class ChatMessage(BaseModel):
    """A chat message with role and content.

    Represents a single message in a conversation. Supports both text-only
    and multimodal (text + images) content for vision language models.

    For vision models (e.g., qwen2.5vl:7b), content can be:
    - A string (text-only, backward compatible)
    - An array of content parts (multimodal):
      - {"type": "text", "text": "..."}
      - {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}

    Attributes:
        role: Message role. Must be "user", "assistant", or "system".
        content: Message content. Can be:
            - str: Text-only content (backward compatible)
            - list[ContentPart]: Multimodal content with text and/or images
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Message role: 'user', 'assistant', or 'system'"
    )
    content: Union[str, list[ContentPart]] = Field(
        ...,
        description=(
            "Message content. Can be a string (text-only) or an array of content parts "
            "(multimodal with text and/or images). For images, use base64-encoded data URLs: "
            "data:image/<format>;base64,<base64_data>"
        ),
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Union[str, list[dict[str, Any]]]) -> Union[str, list[ContentPart]]:
        """Validate and normalize content."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Content cannot be empty")
            return v
        if isinstance(v, list):
            if not v:
                raise ValueError("Content array cannot be empty")
            # Convert dicts to proper models
            parts: list[ContentPart] = []
            for part in v:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        parts.append(TextContentPart(**part))
                    elif part_type == "image_url":
                        parts.append(ImageContentPart(**part))
                    else:
                        raise ValueError(f"Unknown content part type: {part_type}")
                elif isinstance(part, (TextContentPart, ImageContentPart)):
                    parts.append(part)
                else:
                    raise ValueError(f"Invalid content part type: {type(part)}")
            return parts
        raise ValueError("Content must be a string or array of content parts")


class ChatRequest(BaseModel):
    """Request model for chat completion endpoint.

    Validates and deserializes POST /api/v1/chat request bodies.

    Attributes:
        messages: List of chat messages. Required, must contain at least one.
        model: Model name. Optional, defaults to service default.
        stream: Whether to stream the response. Defaults to False.
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

    messages: list[ChatMessage] = Field(
        ..., min_length=1, description="List of chat messages"
    )
    model: str | None = Field(None, description="Model to use (defaults to service default)")
    stream: bool = Field(False, description="Whether to stream the response")
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
    prompt_eval_count: int | None = Field(None, ge=0, description="Number of prompt tokens evaluated")
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
    prompt_eval_count: int | None = Field(None, ge=0, description="Number of prompt tokens evaluated")
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
    prompt_eval_count: int | None = Field(None, ge=0, description="Number of prompt tokens evaluated")
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
    prompt_eval_count: int | None = Field(None, ge=0, description="Number of prompt tokens evaluated")
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
        name: Model name (e.g., "qwen2.5vl:7b").
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
    rejected: int = Field(..., ge=0, description="Total requests rejected (queue full) since startup")
    timeout: int = Field(..., ge=0, description="Total requests timed out waiting in queue since startup")
    total_wait_time_ms: float = Field(..., ge=0.0, description="Total time all requests spent waiting (ms)")
    max_wait_time_ms: float = Field(..., ge=0.0, description="Maximum wait time observed (ms)")
    avg_wait_time_ms: float = Field(..., ge=0.0, description="Average wait time per request (ms)")
    max_concurrent: int = Field(..., ge=1, description="Maximum concurrent requests allowed")
    max_queue_size: int = Field(..., ge=1, description="Maximum queue size")
    default_timeout: float = Field(..., ge=0.0, description="Default queue timeout (seconds)")


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
