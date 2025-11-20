"""Synchronous client for interacting with the shared Ollama service.

This module provides a synchronous HTTP client for the Ollama service using
the requests library. It handles connection pooling, retries, metrics collection,
and structured logging.

Key behaviors:
    - Uses requests.Session with HTTPAdapter for connection pooling
    - Automatic retry logic via HTTPAdapter (max 3 retries)
    - Comprehensive metrics and logging for all operations
    - Supports text generation, chat completion, and model management
    - Validates responses and handles errors gracefully

Thread safety:
    - Each SharedOllamaClient instance uses its own session
    - Safe for concurrent use from multiple threads (each thread should
      use its own client instance)
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from shared_ollama.telemetry.metrics import MetricsCollector
from shared_ollama.telemetry.structured_logging import log_request_event

logger = logging.getLogger(__name__)


def _serialize_options(options: GenerateOptions | dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize generation options into a plain dict."""
    if options is None:
        return None
    if isinstance(options, dict):
        return {k: v for k, v in options.items() if v is not None}

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


class Model(StrEnum):
    """Supported Qwen 3 model identifiers for the client SDK."""

    QWEN3_VL_8B_Q4 = "qwen3-vl:8b-instruct-q4_K_M"
    QWEN3_14B_Q4 = "qwen3:14b-q4_K_M"
    QWEN3_VL_32B = "qwen3-vl:32b"
    QWEN3_30B = "qwen3:30b"


_DEFAULT_CLIENT_MODEL = os.getenv(
    "OLLAMA_DEFAULT_VLM_MODEL",
    Model.QWEN3_VL_8B_Q4.value,
)


@dataclass(slots=True, frozen=True)
class OllamaConfig:
    """Configuration for the synchronous Ollama client.

    Immutable configuration object. All time values are in seconds.

    Attributes:
        base_url: Base URL for Ollama service (default: "http://localhost:11434").
        default_model: Default model to use if not specified in requests
            (default: OLLAMA_DEFAULT_VLM_MODEL env var or Model.QWEN3_VL_8B_Q4).
        timeout: Request timeout in seconds for long operations like generation
            (default: 300).
        health_check_timeout: Timeout for quick health checks in seconds
            (default: 5).
        verbose: Whether to enable verbose logging (default: False).
    """

    base_url: str = "http://localhost:11434"
    default_model: str = _DEFAULT_CLIENT_MODEL
    timeout: int = 300
    health_check_timeout: int = 5
    verbose: bool = False


@dataclass(slots=True, frozen=True)
class GenerateOptions:
    """Options for text generation.

    Immutable configuration object for generation parameters.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Lower values make output
            more deterministic (default: 0.2).
        top_p: Nucleus sampling parameter (0.0-1.0) (default: 0.9).
        top_k: Top-k sampling parameter. Number of tokens to consider
            (default: 40).
        repeat_penalty: Penalty for repetition (default: 1.1).
        max_tokens: Maximum tokens to generate. None means no limit.
        seed: Random seed for reproducibility. None means random.
        stop: List of stop sequences. Generation stops when any sequence
            is encountered.
    """

    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None


@dataclass(slots=True)
class GenerateResponse:
    """Response from Ollama generate API.

    Contains the generated text and performance metrics. All duration values
    are in nanoseconds.

    Attributes:
        text: Generated text content.
        model: Model name used for generation.
        context: Context tokens for continuation (optional).
        total_duration: Total generation duration in nanoseconds.
        load_duration: Model load time in nanoseconds (0 if already loaded).
        prompt_eval_count: Number of prompt tokens evaluated.
        prompt_eval_duration: Prompt evaluation time in nanoseconds.
        eval_count: Number of generation tokens produced.
        eval_duration: Generation time in nanoseconds.
    """

    text: str
    model: str
    context: list[int] | None = None
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class SharedOllamaClient:
    """Unified Ollama client for all projects (synchronous version).

    Provides a synchronous interface to the Ollama service with connection
    pooling, automatic retries, and comprehensive observability.

    Attributes:
        config: Client configuration (OllamaConfig).
        session: requests.Session instance with connection pooling configured.

    Thread safety:
        Safe for concurrent use from multiple threads. Each thread should
        ideally use its own client instance for best performance.
    """

    __slots__ = ("config", "session")

    def __init__(
        self, config: OllamaConfig | None = None, verify_on_init: bool = True
    ) -> None:
        """Initialize synchronous Ollama client.

        Sets up connection pooling and optionally verifies connection to
        the service.

        Args:
            config: Client configuration. If None, uses default OllamaConfig().
            verify_on_init: If True, performs health check during initialization.
                Raises ConnectionError if service is unavailable.

        Raises:
            ConnectionError: If verify_on_init is True and service is unavailable.
        """
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        if verify_on_init:
            self._verify_connection()

    def _verify_connection(self, retries: int = 3, delay: float = 1.0) -> None:
        """Verify connection to Ollama service with retries.

        Performs a health check by requesting /api/tags endpoint. Retries
        on failure with exponential backoff.

        Args:
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Raises:
            ConnectionError: If connection cannot be established after all
                retries. Includes helpful error message with instructions.

        Side effects:
            Makes HTTP GET request to /api/tags endpoint.
        """
        for attempt in range(retries):
            try:
                response = self.session.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=self.config.health_check_timeout,
                )
                response.raise_for_status()
                logger.info("Connected to Ollama service")
                return
            except requests.exceptions.RequestException as exc:
                if attempt < retries - 1:
                    logger.warning(
                        "Connection attempt %s failed, retrying in %ss...",
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.exception("Failed to connect to Ollama after %s attempts", retries)
                    msg = (
                        f"Cannot connect to Ollama at {self.config.base_url}. "
                        "Make sure the service is running.\n"
                        "Start with: ./scripts/start.sh (REST API manages Ollama internally)"
                    )
                    raise ConnectionError(msg) from exc

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models.

        Retrieves the list of models available in the Ollama service.

        Returns:
            List of model dictionaries. Each dictionary contains model metadata
            (name, size, modified_at, etc.).

        Raises:
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            requests.exceptions.HTTPError: If HTTP request fails.

        Side effects:
            - Makes HTTP GET request to /api/tags
            - Logs request event
            - Records metrics
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            data = response.json()
            match isinstance(data, dict):
                case True:
                    return data.get("models", [])
                case False:
                    msg = f"Expected dict response, got {type(data).__name__}"
                    raise ValueError(msg)

        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON response from /api/tags")
            raise
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else None
            logger.exception(
                "HTTP error listing models: %s", status_code or "unknown"
            )
            raise

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
        format: str | dict[str, Any] | None = None,
    ) -> GenerateResponse:
        """Generate text from a prompt.

        Sends a text generation request to the Ollama service and returns
        the generated text with performance metrics.

        Args:
            prompt: Text prompt for generation. Must not be empty.
            model: Model name. If None, uses config.default_model.
            system: Optional system message to set model behavior.
            options: Generation options (temperature, top_p, etc.). If None,
                uses model defaults.
            stream: Whether to stream the response. Note: streaming is not
                implemented in the sync client - use async_client for streaming.
            format: Output format specification. Can be:
                - "json" for JSON mode
                - dict with JSON schema for structured output
                - None for default text output

        Returns:
            GenerateResponse with generated text and performance metrics.

        Raises:
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            requests.exceptions.HTTPError: If HTTP request fails.
            requests.exceptions.RequestException: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/generate
            - Logs request event with metrics
            - Records metrics via MetricsCollector
        """
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

        options_dict = _serialize_options(options)
        if options_dict:
            payload["options"] = options_dict

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.config.base_url}/api/generate",
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
                    "client_type": "sync",
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
        except requests.exceptions.HTTPError as exc:
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
        except requests.exceptions.RequestException as exc:
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
            "client_type": "sync",
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

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        stream: bool = False,
        options: GenerateOptions | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chat completion with multiple messages.

        Sends a chat completion request with a conversation history.
        Supports both text-only and multimodal (text + images) content.

        Args:
            messages: List of chat messages. Each message must be a dict with
                'role' ('user', 'assistant', or 'system') and 'content' keys.
                Content can be:
                - str: Text-only content (backward compatible)
                - list[dict]: Multimodal content with parts:
                  - {"type": "text", "text": "..."}
                  - {"type": "image_url", "image_url": {"url": "data:image/..."}}
            model: Model name. If None, uses config.default_model.
            stream: Whether to stream the response. Note: streaming is not
                implemented in the sync client.
            options: Optional generation options controlling temperature, etc.

        Returns:
            Dictionary with chat response. Contains 'message' dict with 'role'
            and 'content', plus metadata (model, eval_count, etc.).

        Raises:
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            requests.exceptions.HTTPError: If HTTP request fails.
            requests.exceptions.RequestException: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/chat
            - Logs request event with metrics
            - Records metrics via MetricsCollector
        """
        model_str = str(model or self.config.default_model)

        payload: dict[str, Any] = {"model": model_str, "messages": messages, "stream": stream}
        options_dict = _serialize_options(options)
        if options_dict:
            payload["options"] = options_dict
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.config.base_url}/api/chat",
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
                    "client_type": "sync",
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
        except requests.exceptions.HTTPError as exc:
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
        except requests.exceptions.RequestException as exc:
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
            "client_type": "sync",
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

    def pull_model(self, model: str) -> dict[str, Any]:
        """Pull/download a model from Ollama registry.

        Downloads a model from the Ollama registry if not already present.
        This is a long-running operation that may take several minutes.

        Args:
            model: Model name to pull (e.g., "qwen3-vl:8b-instruct-q4_K_M").

        Returns:
            Dictionary with pull response and status information.

        Raises:
            ValueError: If response is not a dictionary.
            json.JSONDecodeError: If response is not valid JSON.
            requests.exceptions.HTTPError: If HTTP request fails.
            requests.exceptions.RequestException: If network error occurs.

        Side effects:
            - Makes HTTP POST request to /api/pull
            - Downloads model files (may take several minutes)
            - Logs request event
        """
        payload = {"name": model}
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json=payload,
                timeout=300,
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
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "pull",
                    "status": "success",
                    "model": model,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "response": data,
                }
            )

            return data

        except json.JSONDecodeError as exc:
            self._log_pull_error(model, request_id, start_time, "JSONDecodeError", str(exc))
            logger.exception("Failed to decode JSON response from /api/pull for %s", model)
            raise
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else None
            self._log_pull_error(model, request_id, start_time, "HTTPError", str(exc), status_code)
            logger.exception("HTTP error pulling model %s: %s", model, status_code)
            raise
        except requests.exceptions.RequestException as exc:
            self._log_pull_error(model, request_id, start_time, exc.__class__.__name__, str(exc))
            logger.exception("Request error pulling model %s: %s", model, exc)
            raise

    def _log_pull_error(
        self,
        model: str,
        request_id: str,
        start_time: float,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
    ) -> None:
        """Log pull error with consistent format.

        Helper method to reduce code duplication in error handling.

        Args:
            model: Model name being pulled.
            request_id: Unique request identifier.
            start_time: Request start time (perf_counter).
            error_type: Type of error that occurred.
            error_message: Error message string.
            status_code: Optional HTTP status code if applicable.

        Side effects:
            - Logs structured event via log_request_event
        """
        latency_ms = (time.perf_counter() - start_time) * 1000

        log_data: dict[str, Any] = {
            "event": "ollama_request",
            "client_type": "sync",
            "operation": "pull",
            "status": "error",
            "model": model,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 3),
            "error_type": error_type,
            "error_message": error_message,
        }
        if status_code is not None:
            log_data["http_status"] = status_code

        log_request_event(log_data)

    def health_check(self) -> bool:
        """Perform health check on Ollama service.

        Performs a lightweight health check by requesting /api/tags endpoint.

        Returns:
            True if service responds with HTTP 200, False otherwise.

        Side effects:
            Makes HTTP GET request to /api/tags endpoint.
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/tags", timeout=5
            )
            match response.status_code:
                case HTTPStatus.OK:
                    return True
                case _:
                    return False
        except requests.exceptions.RequestException as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    @functools.lru_cache(maxsize=128)
    def get_model_info(self, model: str) -> dict[str, Any] | None:
        """Get information about a specific model.

        Searches the list of available models and returns metadata for the
        specified model. Results are cached for performance.

        Args:
            model: Model name to look up.

        Returns:
            Model dictionary if found, None otherwise. Dictionary contains
            model metadata (name, size, modified_at, etc.).

        Side effects:
            - Calls list_models() if cache miss
            - Caches result for subsequent calls
        """
        models = self.list_models()
        return next((item for item in models if item.get("name") == model), None)


__all__ = [
    "GenerateOptions",
    "GenerateResponse",
    "Model",
    "OllamaConfig",
    "SharedOllamaClient",
]
