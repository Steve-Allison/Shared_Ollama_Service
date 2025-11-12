"""
Unified Ollama Client Library
=============================

A common Ollama client that all projects can use to interact with the shared Ollama service.

Usage:
    from shared_ollama_client import SharedOllamaClient

    client = SharedOllamaClient()
    response = client.generate("Hello, world!")
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from monitoring import MetricsCollector
from structured_logging import log_request_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(StrEnum):
    """Available Ollama models."""

    QWEN25_VL_7B = "qwen2.5vl:7b"  # Primary: 7B params, vision-language model
    QWEN25_7B = "qwen2.5:7b"  # Standard: 7B params, text-only model
    QWEN25_14B = "qwen2.5:14b"  # Secondary: 14.8B params
    GRANITE_4_H_TINY = "granite4:tiny-h"  # Granite 4.0 H Tiny: 7B total (1B active), hybrid MoE, optimized for RAG and function calling


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    base_url: str = "http://localhost:11434"
    default_model: str = Model.QWEN25_VL_7B
    timeout: int = 300  # 5 minutes for long generations
    health_check_timeout: int = 5  # 5 seconds for quick health checks
    verbose: bool = False


@dataclass
class GenerateOptions:
    """Options for text generation."""

    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None


@dataclass
class GenerateResponse:
    """Response from Ollama generate API."""

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
    """
    Unified Ollama client for all projects.

    This client provides a common interface to the shared Ollama service
    running on port 11434. It supports all standard Ollama operations.

    Example:
        >>> client = SharedOllamaClient()
        >>> response = client.generate("Hello!")
        >>> print(response.text)
    """

    def __init__(self, config: OllamaConfig | None = None, verify_on_init: bool = True):
        """
        Initialize the Ollama client.

        Args:
            config: Optional configuration (uses defaults if not provided)
            verify_on_init: If True, verify connection immediately (default: True)
        """
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        # Configure connection pooling for better performance
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
        """
        Verify connection to Ollama service with retry logic.

        Args:
            retries: Number of retry attempts
            delay: Delay between retries in seconds
        """
        for attempt in range(retries):
            try:
                response = self.session.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=self.config.health_check_timeout,
                )
                response.raise_for_status()
                logger.info("Connected to Ollama service")
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed, retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.exception(f"Failed to connect to Ollama after {retries} attempts")
                    msg = (
                        f"Cannot connect to Ollama at {self.config.base_url}. "
                        "Make sure the service is running.\n"
                        "Start with: ./scripts/setup_launchd.sh or 'ollama serve'"
                    )
                    raise ConnectionError(msg) from e
            else:
                return

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information dictionaries

        Raises:
            requests.exceptions.HTTPError: If HTTP request fails
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If response structure is invalid
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if not isinstance(data, dict):
                msg = f"Expected dict response, got {type(data).__name__}"
                raise ValueError(msg)

            return data.get("models", [])

        except json.JSONDecodeError as e:
            logger.exception("Failed to decode JSON response from /api/tags")
            raise
        except requests.exceptions.HTTPError as e:
            logger.exception(f"HTTP error listing models: {e.response.status_code}")
            raise

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
    ) -> GenerateResponse:
        """
        Generate text using Ollama.

        Args:
            prompt: The prompt to generate from
            model: Model to use (uses default if not provided)
            system: Optional system prompt
            options: Generation options
            stream: Whether to stream the response

        Returns:
            GenerateResponse with generated text and metadata

        Example:
            >>> client = SharedOllamaClient()
            >>> response = client.generate("Why is the sky blue?", model=Model.QWEN25_VL_7B)
            >>> print(response.text)
        """
        model_str = str(model or self.config.default_model)

        payload: dict[str, Any] = {"model": model_str, "prompt": prompt, "stream": stream}
        if system:
            payload["system"] = system

        options_dict: dict[str, Any] | None = None
        if options:
            options_dict = {
                "temperature": options.temperature,
                "top_p": options.top_p,
                "top_k": options.top_k,
                "repeat_penalty": options.repeat_penalty,
            }
            if options.max_tokens:
                options_dict["num_predict"] = options.max_tokens
            if options.seed:
                options_dict["seed"] = options.seed
            if options.stop:
                options_dict["stop"] = options.stop
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
            if not isinstance(data, dict):
                msg = f"Expected dict response, got {type(data).__name__}"
                raise ValueError(msg)

            latency_ms = (time.perf_counter() - start_time) * 1000

            result = GenerateResponse(
                text=data.get("response", ""),
                model=data.get("model", model),
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

        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error="JSONDecodeError",
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "generate",
                    "status": "error",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "JSONDecodeError",
                    "error_message": str(e),
                }
            )
            logger.exception(f"Failed to decode JSON response from /api/generate for {model_str}")
            raise
        except requests.exceptions.HTTPError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = e.response.status_code if e.response is not None else None
            MetricsCollector.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error=f"HTTPError:{status_code}",
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "generate",
                    "status": "error",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "HTTPError",
                    "http_status": status_code,
                    "error_message": str(e),
                }
            )
            logger.exception(f"HTTP error generating with {model_str}: {status_code}")
            raise
        except requests.exceptions.RequestException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error=e.__class__.__name__,
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "generate",
                    "status": "error",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": e.__class__.__name__,
                    "error_message": str(e),
                }
            )
            logger.exception(f"Request error generating with {model_str}: {e}")
            raise

    def chat(
        self, messages: list[dict[str, str]], model: str | None = None, stream: bool = False
    ) -> dict[str, Any]:
        """
        Chat with the model using chat format.

        Args:
            messages: List of message dicts with "role" and "content"
            model: Model to use (uses default if not provided)
            stream: Whether to stream the response

        Returns:
            Chat response dictionary

        Example:
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response = client.chat(messages)
            >>> print(response["message"]["content"])
        """
        model_str = str(model or self.config.default_model)

        payload = {"model": model_str, "messages": messages, "stream": stream}
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
            if not isinstance(data, dict):
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

        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                error="JSONDecodeError",
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "chat",
                    "status": "error",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "JSONDecodeError",
                    "error_message": str(e),
                }
            )
            logger.exception(f"Failed to decode JSON response from /api/chat for {model_str}")
            raise
        except requests.exceptions.HTTPError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = e.response.status_code if e.response is not None else None
            MetricsCollector.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                error=f"HTTPError:{status_code}",
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "chat",
                    "status": "error",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "HTTPError",
                    "http_status": status_code,
                    "error_message": str(e),
                }
            )
            logger.exception(f"HTTP error in chat with {model_str}: {status_code}")
            raise
        except requests.exceptions.RequestException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                error=e.__class__.__name__,
            )
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "chat",
                    "status": "error",
                    "model": model_str,
                    "stream": stream,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": e.__class__.__name__,
                    "error_message": str(e),
                }
            )
            logger.exception(f"Request error in chat with {model_str}: {e}")
            raise

    def pull_model(self, model: str) -> dict[str, Any]:
        """
        Pull a model from Ollama.

        Args:
            model: Model name to pull

        Returns:
            Status dictionary
        """
        payload = {"name": model}

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        try:
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json=payload,
                timeout=300,  # Pulling can take a while
            )
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict):
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

        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "pull",
                    "status": "error",
                    "model": model,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "JSONDecodeError",
                    "error_message": str(e),
                }
            )
            logger.exception(f"Failed to decode JSON response from /api/pull for {model}")
            raise
        except requests.exceptions.HTTPError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = e.response.status_code if e.response is not None else None
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "pull",
                    "status": "error",
                    "model": model,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "HTTPError",
                    "http_status": status_code,
                    "error_message": str(e),
                }
            )
            logger.exception(f"HTTP error pulling model {model}: {status_code}")
            raise
        except requests.exceptions.RequestException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            log_request_event(
                {
                    "event": "ollama_request",
                    "client_type": "sync",
                    "operation": "pull",
                    "status": "error",
                    "model": model,
                    "request_id": request_id,
                    "latency_ms": round(latency_ms, 3),
                    "error_type": e.__class__.__name__,
                    "error_message": str(e),
                }
            )
            logger.exception(f"Request error pulling model {model}: {e}")
            raise

    def health_check(self) -> bool:
        """
        Perform health check on Ollama service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.debug(f"Health check failed: {e}")
            return False
        else:
            return response.status_code == HTTPStatus.OK

    def get_model_info(self, model: str) -> dict[str, Any] | None:
        """
        Get information about a specific model.

        Args:
            model: Model name

        Returns:
            Model information dictionary or None if not found
        """
        models = self.list_models()
        for m in models:
            if m.get("name") == model:
                return m
        return None


# Convenience functions for easy usage
def create_client(base_url: str = "http://localhost:11434") -> SharedOllamaClient:
    """Create a shared Ollama client with default config."""
    return SharedOllamaClient(OllamaConfig(base_url=base_url))


def quick_generate(prompt: str, model: str | None = None) -> str:
    """
    Quick generate function for simple use cases.

    Args:
        prompt: The prompt to generate from
        model: Model to use (optional)

    Returns:
        Generated text

    Example:
        >>> text = quick_generate("Hello!")
    """
    client = SharedOllamaClient()
    response = client.generate(prompt, model=model)
    return response.text
