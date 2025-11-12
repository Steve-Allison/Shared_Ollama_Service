"""
Async Unified Ollama Client Library
===================================

Modern async/await client for the shared Ollama service.

Usage:
    import asyncio
    from shared_ollama_client_async import AsyncSharedOllamaClient

    async def main():
        client = AsyncSharedOllamaClient()
        response = await client.generate("Hello, world!")
        print(response.text)

    asyncio.run(main())
"""

import asyncio
import json
import logging
import time
import types
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

try:
    import httpx
except ImportError as e:
    msg = "httpx is required for async support. Install with: pip install httpx"
    raise ImportError(msg) from e

from shared_ollama_client import (
    GenerateOptions,
    GenerateResponse,
    Model,
)

from monitoring import MetricsCollector
from structured_logging import log_request_event

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AsyncOllamaConfig:
    """Configuration for Async Ollama client."""

    base_url: str = "http://localhost:11434"
    default_model: str = Model.QWEN25_VL_7B
    timeout: int = 300  # 5 minutes for long generations
    health_check_timeout: int = 5  # 5 seconds for quick health checks
    verbose: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    max_connections: int = 50
    max_keepalive_connections: int = 20
    max_concurrent_requests: int | None = None
    client_limits: httpx.Limits | None = field(default=None, repr=False)
    client_timeout: httpx.Timeout | None = field(default=None, repr=False)


class AsyncSharedOllamaClient:
    """
    Async Unified Ollama client for all projects.

    This client provides an async interface to the shared Ollama service
    running on port 11434. It supports all standard Ollama operations.

    Example:
        >>> async def main():
        ...     client = AsyncSharedOllamaClient()
        ...     response = await client.generate("Hello!")
        ...     print(response.text)
        >>> asyncio.run(main())
    """

    def __init__(self, config: AsyncOllamaConfig | None = None, verify_on_init: bool = True):
        """
        Initialize the async Ollama client.

        Args:
            config: Optional configuration (uses defaults if not provided)
            verify_on_init: If True, verify connection immediately (default: True)
        """
        self.config = config or AsyncOllamaConfig()
        self.client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        if self.config.max_concurrent_requests:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        if verify_on_init:
            # Note: This will be async, but we can't await in __init__
            # So we'll verify on first use instead
            self._needs_verification = True
        else:
            self._needs_verification = False

    async def __aenter__(self) -> "AsyncSharedOllamaClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
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
    async def _acquire_slot(self):
        """
        Optionally throttle concurrent requests with a semaphore.

        Calling code should use this context manager around network requests
        to avoid overwhelming the Ollama service when concurrency is high.
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
        """
        Verify connection to Ollama service with retry logic.

        Args:
            retries: Number of retry attempts (uses config default if None)
            delay: Delay between retries in seconds (uses config default if None)
        """
        retries = retries or self.config.max_retries
        delay = delay or self.config.retry_delay

        await self._ensure_client()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        for attempt in range(retries):
            try:
                response = await self.client.get(
                    "/api/tags",
                    timeout=self.config.health_check_timeout,
                )
                response.raise_for_status()
                logger.info("Connected to Ollama service")
                self._needs_verification = False
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed, retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
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

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information dictionaries

        Raises:
            httpx.HTTPStatusError: If HTTP request fails
            json.JSONDecodeError: If response is not valid JSON
            ValueError: If response structure is invalid
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            async with self._acquire_slot():
                # Quick API call, use health_check_timeout
                response = await self.client.get(
                    "/api/tags",
                    timeout=self.config.health_check_timeout,
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
        except httpx.HTTPStatusError as e:
            logger.exception(f"HTTP error listing models: {e.response.status_code}")
            raise

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
    ) -> GenerateResponse:
        """
        Generate text using Ollama (async).

        Args:
            prompt: The prompt to generate from
            model: Model to use (uses default if not provided)
            system: Optional system prompt
            options: Generation options
            stream: Whether to stream the response

        Returns:
            GenerateResponse with generated text and metadata

        Example:
            >>> async def main():
            ...     client = AsyncSharedOllamaClient()
            ...     response = await client.generate("Why is the sky blue?")
            ...     print(response.text)
            >>> asyncio.run(main())
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
            async with self._acquire_slot():
                response = await self.client.post(
                    "/api/generate",
                    json=payload,
                )  # Timeout configured in client initialization
                response.raise_for_status()

            data = response.json()

            # Validate response structure
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
                    "client_type": "async",
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
        except httpx.HTTPStatusError as e:
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
                    "client_type": "async",
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
        except httpx.RequestError as e:
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
                    "client_type": "async",
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

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Chat with the model using chat format (async).

        Args:
            messages: List of message dicts with "role" and "content"
            model: Model to use (uses default if not provided)
            stream: Whether to stream the response

        Returns:
            Chat response dictionary

        Example:
            >>> async def main():
            ...     client = AsyncSharedOllamaClient()
            ...     messages = [{"role": "user", "content": "Hello!"}]
            ...     response = await client.chat(messages)
            ...     print(response["message"]["content"])
            >>> asyncio.run(main())
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

            # Validate response structure
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
                    "client_type": "async",
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
        except httpx.HTTPStatusError as e:
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
                    "client_type": "async",
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
        except httpx.RequestError as e:
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
                    "client_type": "async",
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

    async def health_check(self) -> bool:
        """
        Perform health check on Ollama service (async).

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            await self._ensure_client()
            if self.client is None:
                logger.debug("Health check failed: Client not initialized")
                return False
            async with self._acquire_slot():
                response = await self.client.get("/api/tags", timeout=5)
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.debug(f"Health check failed: {e}")
            return False
        else:
            return response.status_code == HTTPStatus.OK

    async def get_model_info(self, model: str) -> dict[str, Any] | None:
        """
        Get information about a specific model (async).

        Args:
            model: Model name

        Returns:
            Model information dictionary or None if not found
        """
        models = await self.list_models()
        for m in models:
            if m.get("name") == model:
                return m
        return None


# Convenience functions for easy usage
def create_async_client(base_url: str = "http://localhost:11434") -> AsyncSharedOllamaClient:
    """Create an async shared Ollama client with default config."""
    return AsyncSharedOllamaClient(AsyncOllamaConfig(base_url=base_url))


async def quick_generate_async(prompt: str, model: str | None = None) -> str:
    """
    Quick generate function for simple use cases (async).

    Args:
        prompt: The prompt to generate from
        model: Model to use (optional)

    Returns:
        Generated text

    Example:
        >>> text = await quick_generate_async("Hello!")
    """
    async with AsyncSharedOllamaClient() as client:
        response = await client.generate(prompt, model=model)
        return response.text
