"""
Request and response models for the REST API.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="The prompt to generate text from")
    model: str | None = Field(None, description="Model to use (defaults to service default)")
    system: str | None = Field(None, description="System message for the model")
    stream: bool = Field(False, description="Whether to stream the response")
    format: str | dict | None = Field(
        None,
        description="Output format: 'json' for JSON mode, or JSON schema object for structured output",
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")


class ChatMessage(BaseModel):
    """A chat message."""

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat completion."""

    messages: list[ChatMessage] = Field(..., description="List of chat messages")
    model: str | None = Field(None, description="Model to use (defaults to service default)")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling parameter")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    stop: list[str] | None = Field(None, description="Stop sequences")


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used for generation")
    request_id: str = Field(..., description="Unique request identifier")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, description="Model load time in milliseconds")
    model_warm_start: bool = Field(..., description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(None, description="Number of prompt tokens evaluated")
    generation_eval_count: int | None = Field(
        None, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, description="Total generation duration in milliseconds"
    )


class ChatResponse(BaseModel):
    """Response model for chat completion."""

    message: ChatMessage = Field(..., description="Assistant's response message")
    model: str = Field(..., description="Model used for generation")
    request_id: str = Field(..., description="Unique request identifier")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, description="Model load time in milliseconds")
    model_warm_start: bool = Field(..., description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(None, description="Number of prompt tokens evaluated")
    generation_eval_count: int | None = Field(
        None, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, description="Total generation duration in milliseconds"
    )


class GenerateStreamChunk(BaseModel):
    """Streaming chunk for generate endpoint."""

    chunk: str = Field(..., description="Incremental text chunk")
    done: bool = Field(..., description="Whether generation is complete")
    model: str | None = Field(None, description="Model name")
    request_id: str | None = Field(None, description="Request ID")
    # Metrics only in final chunk (when done=True)
    latency_ms: float | None = Field(None, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, description="Model load time in milliseconds")
    model_warm_start: bool | None = Field(None, description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(None, description="Number of prompt tokens evaluated")
    generation_eval_count: int | None = Field(
        None, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, description="Total generation duration in milliseconds"
    )


class ChatStreamChunk(BaseModel):
    """Streaming chunk for chat endpoint."""

    chunk: str = Field(..., description="Incremental message content")
    role: str = Field(default="assistant", description="Message role")
    done: bool = Field(..., description="Whether response is complete")
    model: str | None = Field(None, description="Model name")
    request_id: str | None = Field(None, description="Request ID")
    # Metrics only in final chunk (when done=True)
    latency_ms: float | None = Field(None, description="Request latency in milliseconds")
    model_load_ms: float | None = Field(None, description="Model load time in milliseconds")
    model_warm_start: bool | None = Field(None, description="Whether model was already loaded")
    prompt_eval_count: int | None = Field(None, description="Number of prompt tokens evaluated")
    generation_eval_count: int | None = Field(
        None, description="Number of generation tokens evaluated"
    )
    total_duration_ms: float | None = Field(
        None, description="Total generation duration in milliseconds"
    )


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str = Field(..., description="Model name")
    size: int | None = Field(None, description="Model size in bytes")
    modified_at: str | None = Field(None, description="Last modification time")


class ModelsResponse(BaseModel):
    """Response model for listing models."""

    models: list[ModelInfo] = Field(..., description="List of available models")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    ollama_service: str = Field(..., description="Ollama service status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    error_type: str | None = Field(None, description="Error type")
    request_id: str | None = Field(None, description="Request identifier if available")


class QueueStatsResponse(BaseModel):
    """Response model for queue statistics."""

    queued: int = Field(..., description="Number of requests currently waiting in queue")
    in_progress: int = Field(..., description="Number of requests currently being processed")
    completed: int = Field(..., description="Total requests completed since startup")
    failed: int = Field(..., description="Total requests failed since startup")
    rejected: int = Field(..., description="Total requests rejected (queue full) since startup")
    timeout: int = Field(..., description="Total requests timed out waiting in queue since startup")
    total_wait_time_ms: float = Field(..., description="Total time all requests spent waiting (ms)")
    max_wait_time_ms: float = Field(..., description="Maximum wait time observed (ms)")
    avg_wait_time_ms: float = Field(..., description="Average wait time per request (ms)")
    max_concurrent: int = Field(..., description="Maximum concurrent requests allowed")
    max_queue_size: int = Field(..., description="Maximum queue size")
    default_timeout: float = Field(..., description="Default queue timeout (seconds)")


@dataclass
class RequestContext:
    """Context for tracking API requests."""

    request_id: str
    client_ip: str
    user_agent: str | None = None
    project_name: str | None = None
