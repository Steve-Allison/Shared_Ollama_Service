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
import logging
from dataclasses import dataclass
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
        if verify_on_init:
            # Note: This will be async, but we can't await in __init__
            # So we'll verify on first use instead
            self._needs_verification = True
        else:
            self._needs_verification = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=self.config.timeout,
                    write=5.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                ),
            )
            if self._needs_verification:
                await self._verify_connection()

    async def _verify_connection(
        self, retries: int | None = None, delay: float | None = None
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
            except Exception as e:
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

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models.

        Returns:
            List of model information dictionaries
        """
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        # Quick API call, use health_check_timeout
        response = await self.client.get(
            "/api/tags",
            timeout=self.config.health_check_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])

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

        if options:
            options_dict: dict[str, Any] = {
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

        logger.info(f"Generating with model {model_str}")

        response = await self.client.post(
            "/api/generate",
            json=payload,
        )  # Timeout configured in client initialization
        response.raise_for_status()

        data = response.json()

        return GenerateResponse(
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

        response = await self.client.post(
            "/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        return response.json()

    async def health_check(self) -> bool:
        """
        Perform health check on Ollama service (async).

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            await self._ensure_client()
            if self.client is None:
                return False
            response = await self.client.get("/api/tags", timeout=5)
        except Exception:
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
