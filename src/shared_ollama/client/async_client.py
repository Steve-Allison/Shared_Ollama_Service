"""Asynchronous client for interacting with the shared Ollama service.

This module provides an asynchronous HTTP client for the Ollama service using
httpx. It supports connection pooling, streaming responses, semaphore-based
concurrency control, and comprehensive observability.

Key behaviors:
    - Uses httpx.AsyncClient for async HTTP operations
    - Supports streaming responses for real-time text generation
    - Implements semaphore-based concurrency limiting
    - Automatic connection pooling and keep-alive
    - Comprehensive metrics and logging for all operations

Concurrency:
    - All operations are async and safe for concurrent use
    - Optional semaphore limits concurrent requests
    - Connection pooling handles multiple simultaneous requests efficiently

Thread safety:
    - Safe for use from multiple async tasks
    - Each client instance manages its own connection pool
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import types
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    msg = "httpx is required for async support. Install with: pip install httpx"
    raise ImportError(msg) from exc

from shared_ollama.client.sync import GenerateOptions, GenerateResponse, Model
from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.structured_logging import log_request_event

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AsyncOllamaConfig:
    """Configuration for the asynchronous Ollama client.

    Immutable configuration object. All time values are in seconds.

    Attributes:
        base_url: Base URL for Ollama service (default: "http://localhost:11434").
        default_model: Default model to use if not specified (default: OLLAMA_DEFAULT_VLM_MODEL env var or Model.QWEN3_VL_8B_Q4).
        timeout: Request timeout for long operations like generation (default: 300).
        health_check_timeout: Timeout for quick health checks (default: 5).
        verbose: Whether to enable verbose logging (default: False).
        max_retries: Maximum retry attempts for connection verification (default: 3).
        retry_delay: Delay between retries in seconds (default: 1.0).
        max_connections: Maximum number of concurrent connections in pool (default: 50).
        max_keepalive_connections: Maximum keep-alive connections (default: 20).
        max_concurrent_requests: Maximum concurrent requests (None = unlimited).
        client_limits: Custom httpx.Limits instance (None = use defaults).
        client_timeout: Custom httpx.Timeout instance (None = use defaults).
    """

    base_url: str = "http://localhost:11434"
    default_model: str = os.getenv(
        "OLLAMA_DEFAULT_VLM_MODEL",
        Model.QWEN3_VL_8B_Q4.value,
    )
    timeout: int = 300
    health_check_timeout: int = 5
    verbose: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    max_connections: int = 50
    max_keepalive_connections: int = 20
    max_concurrent_requests: int | None = None
    client_limits: httpx.Limits | None = field(default=None, repr=False)
    client_timeout: httpx.Timeout | None = field(default=None, repr=False)


class AsyncSharedOllamaClient:
    """Async unified Ollama client for all projects.

    Provides an asynchronous interface to the Ollama service with connection
    pooling, streaming support, and comprehensive observability. Can be used
    as an async context manager for automatic resource cleanup.

    Attributes:
        config: Client configuration (AsyncOllamaConfig).
        client: httpx.AsyncClient instance (initialized lazily).
        _semaphore: Optional asyncio.Semaphore for concurrency control.
        _needs_verification: Whether to verify connection on initialization.

    Thread safety:
        Safe for concurrent use from multiple async tasks. Each task should
        ideally use its own client instance for best performance.

    Lifecycle:
        - Initialize with __init__() or use as async context manager
        - Client is initialized lazily on first use
        - Call close() or use context manager exit to cleanup
    """

    __slots__ = ("_needs_verification", "_semaphore", "client", "config")

    def __init__(
        self, config: AsyncOllamaConfig | None = None, verify_on_init: bool = True
    ) -> None:
        """Initialize async Ollama client.

        Sets up configuration and optional semaphore for concurrency control.
        Client connection is initialized lazily on first use.

        Args:
            config: Client configuration. If None, uses default AsyncOllamaConfig().
            verify_on_init: If True, verifies connection during initialization.
                Raises ConnectionError if service is unavailable.
        """
        self.config = config or AsyncOllamaConfig()
        self.client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        if self.config.max_concurrent_requests:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._needs_verification = verify_on_init

    async def __aenter__(self) -> AsyncSharedOllamaClient:
        """Async context manager entry.

        Initializes client and returns self for use in async with statement.

        Returns:
            Self (AsyncSharedOllamaClient instance).

        Side effects:
            Calls _ensure_client() to initialize httpx client.
        """
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit.

        Automatically closes client connection on context exit.

        Args:
            exc_type: Exception type if exception occurred, None otherwise.
            exc_val: Exception value if exception occurred, None otherwise.
            exc_tb: Traceback if exception occurred, None otherwise.
        """
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure httpx client is initialized.

        Creates httpx.AsyncClient with optimized settings if not already
        created. Called automatically on first use.

        Side effects:
            - Creates httpx.AsyncClient instance
            - May call _verify_connection() if _needs_verification is True
        """
        if self.client is None:
            timeout = self.config.client_timeout or httpx.Timeout(
                connect=5.0,
                read=self.config.timeout,
                write=5.0,
                pool=5.0,
            )
            limits = self.config.client_limits or httpx.Limits(
                max_keepalive_connections=self.config.max_keepalive_connections,
                max_connections=self.config.max_connections,
            )
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=timeout,
                limits=limits,
            )
            if self._needs_verification:
                await self._verify_connection()

    @asynccontextmanager
    async def _acquire_slot(self) -> AsyncIterator[None]:
        """Acquire semaphore slot for concurrency control.

        Async context manager that acquires a semaphore slot if concurrency
        limiting is enabled. No-op if semaphore is not configured.

        Yields:
            None. The context manager guarantees a slot is acquired while
            the context is active.

        Side effects:
            - Acquires semaphore on entry (if configured)
            - Releases semaphore on exit (if configured)
        """
        if self._semaphore is None:
            yield
            return

        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    async def _verify_connection(
        self,
        retries: int | None = None,
        delay: float | None = None,
    ) -> None:
        """Verify connection to Ollama service with retries.

        Performs a health check by requesting /api/tags endpoint. Retries
        on failure with configurable delay.

        Args:
            retries: Number of retry attempts. If None, uses config.max_retries.
            delay: Delay between retries in seconds. If None, uses config.retry_delay.

        Raises:
            ConnectionError: If connection cannot be established after all
                retries. Includes helpful error message with instructions.
            RuntimeError: If client is not initialized.

        Side effects:
            - Makes HTTP GET request to /api/tags endpoint
            - Sleeps between retry attempts
        """
        retries = retries or self.config.max_retries
        delay = delay or self.config.retry_delay

        await self._ensure_client()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        for attempt in range(retries):
            try:
                async with self._acquire_slot():
                    response = await self.client.get(
                        "/api/tags",
                        timeout=self.config.health_check_timeout,
                    )
                response.raise_for_status()
                logger.info("Connected to Ollama service")
                self._needs_verification = False
                return
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt < retries - 1:
                    logger.warning(
                        "Connection attempt %s failed, retrying in %ss...",
                        attempt + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception("Failed to connect to Ollama after %s attempts", retries)
                    msg = (
                        f"Cannot connect to Ollama at {self.config.base_url}. "
                        "Make sure the service is running.\n"
                        "Start with: ./scripts/start.sh (REST API manages Ollama internally)"
                    )
                    raise ConnectionError(msg) from exc

    async def close(self) -> None:
        """Close the httpx client and cleanup resources.

        Closes the underlying httpx.AsyncClient connection pool. Safe to
        call multiple times. Should be called when done with the client
        to free resources.

        Side effects:
            - Closes httpx.AsyncClient connection pool
            - Sets self.client to None
        """
        if self.client:
            await self.client.aclose()
            self.client = None

    async def list_models(self) -> list[dict[str, Any]]:
        """List all available models.

        Retrieves the list of models available in the Ollama service.

        Returns:
            List of model dictionaries. Each dictionary contains model metadata
            (name, size, modified_at, etc.).

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            httpx.HTTPStatusError: If HTTP request fails.

        Side effects:
            - Makes HTTP GET request to /api/tags
            - May acquire semaphore slot if concurrency limiting enabled
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        start_time = time.perf_counter()

        try:
            async with self._acquire_slot():
                response = await self.client.get(
                    "/api/tags",
                    timeout=self.config.health_check_timeout,
                )
            response.raise_for_status()

            data = response.json()

            match isinstance(data, dict):
                case True:
                    models = data.get("models", [])
                case False:
                    msg = f"Expected dict response, got {type(data).__name__}"
                    raise ValueError(msg)

            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model="system",
                operation="list_models",
                latency_ms=latency_ms,
                success=True,
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "async",
                    "operation": "list_models",
                    "status": "success",
                    "request_id": str(uuid.uuid4()),
                    "latency_ms": round(latency_ms, 3),
                    "models_returned": len(models),
                }
            )

            return models

        except json.JSONDecodeError:
            self._record_list_models_error(start_time, "JSONDecodeError")
            logger.exception("Failed to decode JSON response from /api/tags")
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response else None
            self._record_list_models_error(start_time, f"HTTPError:{status_code}")
            logger.exception("HTTP error listing models: %s", status_code)
            raise
        except httpx.RequestError as exc:
            self._record_list_models_error(start_time, exc.__class__.__name__)
            logger.exception("Request error listing models: %s", exc)
            raise

    def _record_list_models_error(self, start_time: float, error_type: str) -> None:
        """Record metrics/logs for list_models errors."""
        latency_ms = (time.perf_counter() - start_time) * 1000
        MetricsCollector.record_request(
            model="system",
            operation="list_models",
            latency_ms=latency_ms,
            success=False,
            error=error_type,
        )
        log_request_event(
            {
                "event": "ollama_request",
                "client_type": "async",
                "operation": "list_models",
                "status": "error",
                "request_id": str(uuid.uuid4()),
                "latency_ms": round(latency_ms, 3),
                "error_type": error_type,
            }
        )

    def _build_options_dict(self, options: GenerateOptions | None) -> dict[str, Any] | None:
        """Build options dictionary from GenerateOptions.

        Converts GenerateOptions dataclass to dictionary format expected by
        Ollama API. Filters out None values for optional parameters.

        Args:
            options: GenerateOptions instance or None.

        Returns:
            Options dictionary ready for API payload, or None if options is None.
        """
        if not options:
            return None

        options_dict: dict[str, Any] = {
            "temperature": options.temperature,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "repeat_penalty": options.repeat_penalty,
        }

        optional_opts = {
            k: v
            for k, v in {
                "num_predict": options.max_tokens,
                "seed": options.seed,
                "stop": options.stop,
            }.items()
            if v is not None
        }
        options_dict.update(optional_opts)

        return options_dict

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
        format: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateResponse:
        """Generate text from a prompt.

        Sends a text generation request to the Ollama service and returns
        the generated text with performance metrics. Supports tool calling (POML).

        Args:
            prompt: Text prompt for generation. Must not be empty.
            model: Model name. If None, uses config.default_model.
            system: Optional system message to set model behavior.
            options: Generation options (temperature, top_p, etc.). If None,
                uses model defaults.
            stream: Whether to stream the response. Note: use generate_stream()
                method for actual streaming.
            format: Output format specification. Can be:
                - "json" for JSON mode
                - dict with JSON schema for structured output
                - None for default text output
            tools: List of tools/functions the model can call (POML compatible).

        Returns:
            GenerateResponse with generated text and performance metrics.
            May include tool_calls if model calls tools.

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            httpx.HTTPStatusError: If HTTP request fails.
            httpx.RequestError: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/generate
            - Logs request event with metrics
            - Records metrics via MetricsCollector
            - May acquire semaphore slot if concurrency limiting enabled
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        model_str = str(model or self.config.default_model)

        payload: dict[str, Any] = {
            "model": model_str,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        if tools:
            payload["tools"] = tools

        options_dict = self._build_options_dict(options)
        if options_dict:
            payload["options"] = options_dict

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            async with self._acquire_slot():
                response = await self.client.post(
                    "/api/generate",
                    json=payload,
                )
                response.raise_for_status()

            data = response.json()

            match isinstance(data, dict):
                case True:
                    pass
                case False:
                    msg = f"Expected dict response, got {type(data).__name__}"
                    raise ValueError(msg)

            latency_ms = (time.perf_counter() - start_time) * 1000

            result = GenerateResponse(
                text=data.get("response", ""),
                model=data.get("model", model_str),
                context=data.get("context"),
                total_duration=data.get("total_duration", 0),
                load_duration=data.get("load_duration", 0),
                prompt_eval_count=data.get("prompt_eval_count", 0),
                prompt_eval_duration=data.get("prompt_eval_duration", 0),
                eval_count=data.get("eval_count", 0),
                eval_duration=data.get("eval_duration", 0),
            )

            MetricsCollector.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=True,
            )

            load_ms = result.load_duration / 1_000_000 if result.load_duration else 0.0
            total_ms = result.total_duration / 1_000_000 if result.total_duration else 0.0

            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "async",
                    "operation": "generate",
                    "status": "success",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "total_duration_ms": round(total_ms, 3) if total_ms else None,
                    "model_load_ms": round(load_ms, 3) if load_ms else 0.0,
                    "model_warm_start": load_ms == 0.0,
                    "prompt_chars": len(prompt),
                    "prompt_eval_count": result.prompt_eval_count,
                    "generation_eval_count": result.eval_count,
                    "options": options_dict,
                }
            )

            return result

        except json.JSONDecodeError as exc:
            self._log_generate_error(
                model_str, stream, request_id, start_time, "JSONDecodeError", str(exc)
            )
            logger.exception("Failed to decode JSON response from /api/generate for %s", model_str)
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response else None
            self._log_generate_error(
                model_str,
                stream,
                request_id,
                start_time,
                "HTTPError",
                str(exc),
                status_code,
            )
            logger.exception("HTTP error generating with %s: %s", model_str, status_code)
            raise
        except httpx.RequestError as exc:
            self._log_generate_error(
                model_str, stream, request_id, start_time, exc.__class__.__name__, str(exc)
            )
            logger.exception("Request error generating with %s: %s", model_str, exc)
            raise

    def _log_generate_error(
        self,
        model: str,
        stream: bool,
        request_id: str,
        start_time: float,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
    ) -> None:
        """Log generate error with consistent format.

        Helper method to reduce code duplication in error handling.

        Args:
            model: Model name used for the request.
            stream: Whether streaming was requested.
            request_id: Unique request identifier.
            start_time: Request start time (perf_counter).
            error_type: Type of error that occurred.
            error_message: Error message string.
            status_code: Optional HTTP status code if applicable.

        Side effects:
            - Records metrics via MetricsCollector
            - Logs structured event via log_request_event
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_name = f"{error_type}:{status_code}" if status_code else error_type

        MetricsCollector.record_request(
            model=model,
            operation="generate",
            latency_ms=latency_ms,
            success=False,
            error=error_name,
        )

        log_data: dict[str, Any] = {
            "event": "ollama_request",
            "client_type": "async",
            "operation": "generate",
            "status": "error",
            "model": model,
            "stream": stream,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 3),
            "error_type": error_type,
            "error_message": error_message,
        }
        if status_code is not None:
            log_data["http_status"] = status_code

        log_request_event(log_data)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
        images: list[str] | None = None,
        format: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Chat completion with multiple messages.

        Sends a chat completion request with a conversation history.
        Supports text-only, multimodal (text + images), and tool calling (POML).

        Args:
            messages: List of chat messages. Each message must be a dict with
                'role' and optional 'content', 'tool_calls', 'tool_call_id' keys.
                Supports tool calling for POML workflows.
            model: Model name. If None, uses config.default_model.
            options: Generation options (temperature, top_p, etc.). If None,
                uses model defaults.
            stream: Whether to stream the response. Note: use chat_stream()
                method for actual streaming.
            images: List of base64-encoded images (native Ollama format).
            format: Output format. Can be "json" or a JSON schema dict.
            tools: List of tools/functions the model can call (POML compatible).

        Returns:
            Dictionary with chat response. Contains 'message' dict with 'role'
            and 'content', plus metadata (model, eval_count, etc.).
            May include 'tool_calls' in message if model calls tools.

        Raises:
            RuntimeError: If client is not initialized.
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            httpx.HTTPStatusError: If HTTP request fails.
            httpx.RequestError: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/chat
            - Logs request event with metrics
            - Records metrics via MetricsCollector
            - May acquire semaphore slot if concurrency limiting enabled
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        model_str = str(model or self.config.default_model)

        payload: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "stream": stream,
        }

        # Add images parameter if provided (Ollama's native format for vision models)
        if images:
            payload["images"] = images

        # Add format parameter if provided (JSON schema support / POML)
        if format is not None:
            payload["format"] = format

        # Add tools parameter if provided (tool calling / POML)
        if tools:
            payload["tools"] = tools

        options_dict = self._build_options_dict(options)
        if options_dict:
            payload["options"] = options_dict

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            async with self._acquire_slot():
                response = await self.client.post(
                    "/api/chat",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()

            data = response.json()

            match isinstance(data, dict):
                case True:
                    pass
                case False:
                    msg = f"Expected dict response, got {type(data).__name__}"
                    raise ValueError(msg)

            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=True,
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "async",
                    "operation": "chat",
                    "status": "success",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "messages_count": len(messages),
                }
            )

            return data

        except json.JSONDecodeError as exc:
            self._log_chat_error(
                model_str, stream, request_id, start_time, "JSONDecodeError", str(exc)
            )
            logger.exception("Failed to decode JSON response from /api/chat for %s", model_str)
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response else None
            self._log_chat_error(
                model_str,
                stream,
                request_id,
                start_time,
                "HTTPError",
                str(exc),
                status_code,
            )
            logger.exception("HTTP error in chat with %s: %s", model_str, status_code)
            raise
        except httpx.RequestError as exc:
            self._log_chat_error(
                model_str, stream, request_id, start_time, exc.__class__.__name__, str(exc)
            )
            logger.exception("Request error in chat with %s: %s", model_str, exc)
            raise

    def _log_chat_error(
        self,
        model: str,
        stream: bool,
        request_id: str,
        start_time: float,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
    ) -> None:
        """Log chat error with consistent format.

        Helper method to reduce code duplication in error handling.

        Args:
            model: Model name used for the request.
            stream: Whether streaming was requested.
            request_id: Unique request identifier.
            start_time: Request start time (perf_counter).
            error_type: Type of error that occurred.
            error_message: Error message string.
            status_code: Optional HTTP status code if applicable.

        Side effects:
            - Records metrics via MetricsCollector
            - Logs structured event via log_request_event
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_name = f"{error_type}:{status_code}" if status_code else error_type

        MetricsCollector.record_request(
            model=model,
            operation="chat",
            latency_ms=latency_ms,
            success=False,
            error=error_name,
        )

        log_data: dict[str, Any] = {
            "event": "ollama_request",
            "client_type": "async",
            "operation": "chat",
            "status": "error",
            "model": model,
            "stream": stream,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 3),
            "error_type": error_type,
            "error_message": error_message,
        }
        if status_code is not None:
            log_data["http_status"] = status_code

        log_request_event(log_data)

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
        format: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream text generation from Ollama.

        Sends a streaming generation request and yields incremental text chunks
        as they are generated. Final chunk includes complete metrics.
        Supports tool calling (POML).

        Args:
            prompt: Text prompt for generation. Must not be empty.
            model: Model name. If None, uses config.default_model.
            system: Optional system message to set model behavior.
            options: Generation options (temperature, top_p, etc.). If None,
                uses model defaults.
            format: Output format specification. Can be "json" or JSON schema dict.
            tools: List of tools/functions the model can call (POML compatible).

        Yields:
            Dictionary chunks with keys:
                - chunk: str - Incremental text chunk
                - done: bool - Whether generation is complete
                - model: str - Model name
                - request_id: str - Request identifier
            Final chunk (done=True) also includes:
                - latency_ms: float - Total request latency
                - model_load_ms: float - Model load time
                - model_warm_start: bool - Whether model was already loaded
                - prompt_eval_count: int - Prompt tokens evaluated
                - generation_eval_count: int - Generation tokens produced
                - total_duration_ms: float - Total generation duration

        Raises:
            RuntimeError: If client is not initialized.
            httpx.HTTPStatusError: If HTTP request fails.
            httpx.RequestError: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/generate with stream=True
            - Logs request event with metrics on completion
            - Records metrics via MetricsCollector on completion
            - May acquire semaphore slot if concurrency limiting enabled
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        model_str = str(model or self.config.default_model)
        request_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "model": model_str,
            "prompt": prompt,
            "stream": True,
        }

        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        if tools:
            payload["tools"] = tools

        options_dict = self._build_options_dict(options)
        if options_dict:
            payload["options"] = options_dict

        start_time = time.perf_counter()

        try:
            async with self._acquire_slot(), self.client.stream(
                "POST",
                "/api/generate",
                json=payload,
                timeout=self.config.timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse streaming chunk: %s", line)
                        continue

                    done = chunk_data.get("done", False)
                    text_chunk = chunk_data.get("response", "")

                    chunk_dict: dict[str, Any] = {
                        "chunk": text_chunk,
                        "done": done,
                        "model": chunk_data.get("model", model_str),
                        "request_id": request_id,
                    }

                    if done:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        load_duration = chunk_data.get("load_duration", 0)
                        total_duration = chunk_data.get("total_duration", 0)

                        chunk_dict.update(
                            {
                                "latency_ms": round(latency_ms, 3),
                                "model_load_ms": round(load_duration / 1_000_000, 3)
                                if load_duration
                                else 0.0,
                                "model_warm_start": load_duration == 0,
                                "prompt_eval_count": chunk_data.get("prompt_eval_count"),
                                "generation_eval_count": chunk_data.get("eval_count"),
                                "total_duration_ms": round(total_duration / 1_000_000, 3)
                                if total_duration
                                else None,
                            }
                        )

                        MetricsCollector.record_request(
                            model=model_str,
                            operation="generate_stream",
                            latency_ms=latency_ms,
                            success=True,
                        )

                        log_request_event(
                            {
                                "event": "ollama_request",
                                "client_type": "async",
                                "operation": "generate_stream",
                                "status": "success",
                                "model": model_str,
                                "stream": True,
                                "request_id": request_id,
                                "latency_ms": round(latency_ms, 3),
                                "total_duration_ms": chunk_dict.get("total_duration_ms"),
                                "model_load_ms": chunk_dict.get("model_load_ms"),
                                "model_warm_start": chunk_dict.get("model_warm_start"),
                                "prompt_chars": len(prompt),
                                "prompt_eval_count": chunk_dict.get("prompt_eval_count"),
                                "generation_eval_count": chunk_dict.get("generation_eval_count"),
                                "options": options_dict,
                            }
                        )

                    yield chunk_dict

        except httpx.HTTPStatusError as exc:
            self._log_stream_error(
                model_str,
                request_id,
                start_time,
                "generate_stream",
                "HTTPError",
                str(exc),
                exc.response.status_code if exc.response else None,
            )
            logger.exception("HTTP error streaming generate with %s", model_str)
            raise
        except httpx.RequestError as exc:
            self._log_stream_error(
                model_str, request_id, start_time, "generate_stream", exc.__class__.__name__, str(exc)
            )
            logger.exception("Request error streaming generate with %s", model_str)
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        options: GenerateOptions | None = None,
        images: list[str] | None = None,
        format: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completion from Ollama.

        Sends a streaming chat completion request and yields incremental message
        chunks as they are generated. Final chunk includes complete metrics.
        Supports text-only, multimodal (text + images), and tool calling (POML).

        Args:
            messages: List of chat messages. Each message must be a dict with
                'role' and optional 'content', 'tool_calls', 'tool_call_id' keys.
                Supports tool calling for POML workflows.
            model: Model name. If None, uses config.default_model.
            options: Generation options (temperature, top_p, etc.). If None,
                uses model defaults.
            images: List of base64-encoded images (native Ollama format).
            format: Output format. Can be "json" or a JSON schema dict.
            tools: List of tools/functions the model can call (POML compatible).

        Yields:
            Dictionary chunks with keys:
                - chunk: str - Incremental message content
                - role: str - Message role (default: "assistant")
                - done: bool - Whether response is complete
                - model: str - Model name
                - request_id: str - Request identifier
            Final chunk (done=True) also includes:
                - latency_ms: float - Total request latency
                - model_load_ms: float - Model load time
                - model_warm_start: bool - Whether model was already loaded
                - prompt_eval_count: int - Prompt tokens evaluated
                - generation_eval_count: int - Generation tokens produced
                - total_duration_ms: float - Total generation duration

        Raises:
            RuntimeError: If client is not initialized.
            httpx.HTTPStatusError: If HTTP request fails.
            httpx.RequestError: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/chat with stream=True
            - Logs request event with metrics on completion
            - Records metrics via MetricsCollector on completion
            - May acquire semaphore slot if concurrency limiting enabled
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        model_str = str(model or self.config.default_model)
        request_id = str(uuid.uuid4())

        payload: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "stream": True,
        }

        # Add images parameter if provided (Ollama's native format for vision models)
        if images:
            payload["images"] = images

        # Add format parameter if provided (JSON schema support / POML)
        if format is not None:
            payload["format"] = format

        # Add tools parameter if provided (tool calling / POML)
        if tools:
            payload["tools"] = tools

        options_dict = self._build_options_dict(options)
        if options_dict:
            payload["options"] = options_dict

        start_time = time.perf_counter()

        try:
            async with self._acquire_slot(), self.client.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=self.config.timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse streaming chunk: %s", line)
                        continue

                    done = chunk_data.get("done", False)
                    message = chunk_data.get("message", {})
                    text_chunk = message.get("content", "")
                    role = message.get("role", "assistant")

                    chunk_dict: dict[str, Any] = {
                        "chunk": text_chunk,
                        "role": role,
                        "done": done,
                        "model": chunk_data.get("model", model_str),
                        "request_id": request_id,
                    }

                    if done:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        load_duration = chunk_data.get("load_duration", 0)
                        total_duration = chunk_data.get("total_duration", 0)

                        chunk_dict.update(
                            {
                                "latency_ms": round(latency_ms, 3),
                                "model_load_ms": round(load_duration / 1_000_000, 3)
                                if load_duration
                                else 0.0,
                                "model_warm_start": load_duration == 0,
                                "prompt_eval_count": chunk_data.get("prompt_eval_count"),
                                "generation_eval_count": chunk_data.get("eval_count"),
                                "total_duration_ms": round(total_duration / 1_000_000, 3)
                                if total_duration
                                else None,
                            }
                        )

                        MetricsCollector.record_request(
                            model=model_str,
                            operation="chat_stream",
                            latency_ms=latency_ms,
                            success=True,
                        )

                        log_request_event(
                            {
                                "event": "ollama_request",
                                "client_type": "async",
                                "operation": "chat_stream",
                                "status": "success",
                                "model": model_str,
                                "stream": True,
                                "request_id": request_id,
                                "latency_ms": round(latency_ms, 3),
                                "total_duration_ms": chunk_dict.get("total_duration_ms"),
                                "model_load_ms": chunk_dict.get("model_load_ms"),
                                "model_warm_start": chunk_dict.get("model_warm_start"),
                                "messages_count": len(messages),
                                "prompt_eval_count": chunk_dict.get("prompt_eval_count"),
                                "generation_eval_count": chunk_dict.get("generation_eval_count"),
                                "options": options_dict,
                            }
                        )

                    yield chunk_dict

        except httpx.HTTPStatusError as exc:
            self._log_stream_error(
                model_str,
                request_id,
                start_time,
                "chat_stream",
                "HTTPError",
                str(exc),
                exc.response.status_code if exc.response else None,
            )
            logger.exception("HTTP error streaming chat with %s", model_str)
            raise
        except httpx.RequestError as exc:
            self._log_stream_error(
                model_str, request_id, start_time, "chat_stream", exc.__class__.__name__, str(exc)
            )
            logger.exception("Request error streaming chat with %s", model_str)
            raise

    def _log_stream_error(
        self,
        model: str,
        request_id: str,
        start_time: float,
        operation: str,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
    ) -> None:
        """Log streaming error with consistent format.

        Helper method to reduce code duplication in error handling for
        streaming operations.

        Args:
            model: Model name used for the request.
            request_id: Unique request identifier.
            start_time: Request start time (perf_counter).
            operation: Operation name (e.g., "generate_stream", "chat_stream").
            error_type: Type of error that occurred.
            error_message: Error message string.
            status_code: Optional HTTP status code if applicable.

        Side effects:
            - Records metrics via MetricsCollector
            - Logs structured event via log_request_event
        """
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_name = f"{error_type}:{status_code}" if status_code else error_type

        MetricsCollector.record_request(
            model=model,
            operation=operation,
            latency_ms=latency_ms,
            success=False,
            error=error_name,
        )

        log_data: dict[str, Any] = {
            "event": "ollama_request",
            "client_type": "async",
            "operation": operation,
            "status": "error",
            "model": model,
            "stream": True,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 3),
            "error_type": error_type,
            "error_message": error_message,
        }
        if status_code is not None:
            log_data["http_status"] = status_code

        log_request_event(log_data)

    async def health_check(self) -> bool:
        """Perform health check on Ollama service.

        Performs a lightweight health check by requesting /api/tags endpoint.

        Returns:
            True if service responds with HTTP 200, False otherwise.

        Side effects:
            - Makes HTTP GET request to /api/tags endpoint
            - May acquire semaphore slot if concurrency limiting enabled
        """
        try:
            await self._ensure_client()
            if self.client is None:
                logger.debug("Health check failed: Client not initialized")
                return False
            async with self._acquire_slot():
                response = await self.client.get("/api/tags", timeout=5)
            match response.status_code:
                case HTTPStatus.OK:
                    return True
                case _:
                    return False
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    async def get_model_info(self, model: str) -> dict[str, Any] | None:
        """Get information about a specific model.

        Searches the list of available models and returns metadata for the
        specified model.

        Args:
            model: Model name to look up.

        Returns:
            Model dictionary if found, None otherwise. Dictionary contains
            model metadata (name, size, modified_at, etc.).

        Raises:
            RuntimeError: If client is not initialized.
            httpx.HTTPStatusError: If HTTP request fails.

        Side effects:
            - Calls list_models() which makes HTTP request
        """
        models = await self.list_models()
        return next((item for item in models if item.get("name") == model), None)


__all__ = ["AsyncOllamaConfig", "AsyncSharedOllamaClient"]
