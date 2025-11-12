"""
Asynchronous client for interacting with the shared Ollama service.
"""

from __future__ import annotations

import asyncio
import json
import logging
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


@dataclass
class AsyncOllamaConfig:
    """Configuration for the asynchronous Ollama client."""

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
    """Async unified Ollama client for all projects."""

    def __init__(self, config: AsyncOllamaConfig | None = None, verify_on_init: bool = True):
        self.config = config or AsyncOllamaConfig()
        self.client: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        if self.config.max_concurrent_requests:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._needs_verification = verify_on_init

    async def __aenter__(self) -> "AsyncSharedOllamaClient":
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.close()

    async def _ensure_client(self) -> None:
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
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if attempt < retries - 1:
                    logger.warning("Connection attempt %s failed, retrying in %ss...", attempt + 1, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.exception("Failed to connect to Ollama after %s attempts", retries)
                    msg = (
                        f"Cannot connect to Ollama at {self.config.base_url}. "
                        "Make sure the service is running.\n"
                        "Start with: ./scripts/setup_launchd.sh or 'ollama serve'"
                    )
                    raise ConnectionError(msg) from exc
            else:
                return

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def list_models(self) -> list[dict[str, Any]]:
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            async with self._acquire_slot():
                response = await self.client.get(
                    "/api/tags",
                    timeout=self.config.health_check_timeout,
                )
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, dict):
                msg = f"Expected dict response, got {type(data).__name__}"
                raise ValueError(msg)

            return data.get("models", [])

        except json.JSONDecodeError as exc:
            logger.exception("Failed to decode JSON response from /api/tags")
            raise
        except httpx.HTTPStatusError as exc:
            logger.exception("HTTP error listing models: %s", exc.response.status_code)
            raise

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
    ) -> GenerateResponse:
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
                    "error_message": str(exc),
                }
            )
            logger.exception("Failed to decode JSON response from /api/generate for %s", model_str)
            raise
        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = exc.response.status_code if exc.response is not None else None
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
                    "error_message": str(exc),
                }
            )
            logger.exception("HTTP error generating with %s: %s", model_str, status_code)
            raise
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error=exc.__class__.__name__,
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
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            logger.exception("Request error generating with %s: %s", model_str, exc)
            raise

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: GenerateOptions | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("Client not initialized")

        model_str = str(model or self.config.default_model)

        payload: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "stream": stream,
        }

        # Add options if provided (same pattern as generate())
        if options is not None:
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
                    "/api/chat",
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
                    "error_message": str(exc),
                }
            )
            logger.exception("Failed to decode JSON response from /api/chat for %s", model_str)
            raise
        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = exc.response.status_code if exc.response is not None else None
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
                    "error_message": str(exc),
                }
            )
            logger.exception("HTTP error in chat with %s: %s", model_str, status_code)
            raise
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                error=exc.__class__.__name__,
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
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            logger.exception("Request error in chat with %s: %s", model_str, exc)
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream text generation from Ollama.

        Yields GenerateStreamChunk objects with incremental text and metrics.
        Final chunk (done=True) includes complete metrics.
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

        start_time = time.perf_counter()

        try:
            async with self._acquire_slot():
                async with self.client.stream(
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

                        # Prepare chunk with basic info
                        chunk_dict = {
                            "chunk": text_chunk,
                            "done": done,
                            "model": chunk_data.get("model", model_str),
                            "request_id": request_id,
                        }

                        # Add metrics to final chunk
                        if done:
                            latency_ms = (time.perf_counter() - start_time) * 1000
                            load_duration = chunk_data.get("load_duration", 0)
                            total_duration = chunk_data.get("total_duration", 0)

                            chunk_dict.update({
                                "latency_ms": round(latency_ms, 3),
                                "model_load_ms": round(load_duration / 1_000_000, 3) if load_duration else 0.0,
                                "model_warm_start": load_duration == 0,
                                "prompt_eval_count": chunk_data.get("prompt_eval_count"),
                                "generation_eval_count": chunk_data.get("eval_count"),
                                "total_duration_ms": round(total_duration / 1_000_000, 3) if total_duration else None,
                            })

                            MetricsCollector.record_request(
                                model=model_str,
                                operation="generate_stream",
                                latency_ms=latency_ms,
                                success=True,
                            )

                            log_request_event({
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
                            })

                        yield chunk_dict

        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = exc.response.status_code if exc.response is not None else None
            MetricsCollector.record_request(
                model=model_str,
                operation="generate_stream",
                latency_ms=latency_ms,
                success=False,
                error=f"HTTPError:{status_code}",
            )
            log_request_event({
                "event": "ollama_request",
                "client_type": "async",
                "operation": "generate_stream",
                "status": "error",
                "model": model_str,
                "stream": True,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 3),
                "error_type": "HTTPError",
                "http_status": status_code,
                "error_message": str(exc),
            })
            logger.exception("HTTP error streaming generate with %s: %s", model_str, status_code)
            raise
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="generate_stream",
                latency_ms=latency_ms,
                success=False,
                error=exc.__class__.__name__,
            )
            log_request_event({
                "event": "ollama_request",
                "client_type": "async",
                "operation": "generate_stream",
                "status": "error",
                "model": model_str,
                "stream": True,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 3),
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            })
            logger.exception("Request error streaming generate with %s: %s", model_str, exc)
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream chat completion from Ollama.

        Yields ChatStreamChunk objects with incremental message content and metrics.
        Final chunk (done=True) includes complete metrics.
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

        options_dict: dict[str, Any] | None = None
        if options is not None:
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

        start_time = time.perf_counter()

        try:
            async with self._acquire_slot():
                async with self.client.stream(
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

                        # Prepare chunk with basic info
                        chunk_dict = {
                            "chunk": text_chunk,
                            "role": role,
                            "done": done,
                            "model": chunk_data.get("model", model_str),
                            "request_id": request_id,
                        }

                        # Add metrics to final chunk
                        if done:
                            latency_ms = (time.perf_counter() - start_time) * 1000
                            load_duration = chunk_data.get("load_duration", 0)
                            total_duration = chunk_data.get("total_duration", 0)

                            chunk_dict.update({
                                "latency_ms": round(latency_ms, 3),
                                "model_load_ms": round(load_duration / 1_000_000, 3) if load_duration else 0.0,
                                "model_warm_start": load_duration == 0,
                                "prompt_eval_count": chunk_data.get("prompt_eval_count"),
                                "generation_eval_count": chunk_data.get("eval_count"),
                                "total_duration_ms": round(total_duration / 1_000_000, 3) if total_duration else None,
                            })

                            MetricsCollector.record_request(
                                model=model_str,
                                operation="chat_stream",
                                latency_ms=latency_ms,
                                success=True,
                            )

                            log_request_event({
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
                            })

                        yield chunk_dict

        except httpx.HTTPStatusError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            status_code = exc.response.status_code if exc.response is not None else None
            MetricsCollector.record_request(
                model=model_str,
                operation="chat_stream",
                latency_ms=latency_ms,
                success=False,
                error=f"HTTPError:{status_code}",
            )
            log_request_event({
                "event": "ollama_request",
                "client_type": "async",
                "operation": "chat_stream",
                "status": "error",
                "model": model_str,
                "stream": True,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 3),
                "error_type": "HTTPError",
                "http_status": status_code,
                "error_message": str(exc),
            })
            logger.exception("HTTP error streaming chat with %s: %s", model_str, status_code)
            raise
        except httpx.RequestError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            MetricsCollector.record_request(
                model=model_str,
                operation="chat_stream",
                latency_ms=latency_ms,
                success=False,
                error=exc.__class__.__name__,
            )
            log_request_event({
                "event": "ollama_request",
                "client_type": "async",
                "operation": "chat_stream",
                "status": "error",
                "model": model_str,
                "stream": True,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 3),
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            })
            logger.exception("Request error streaming chat with %s: %s", model_str, exc)
            raise

    async def health_check(self) -> bool:
        try:
            await self._ensure_client()
            if self.client is None:
                logger.debug("Health check failed: Client not initialized")
                return False
            async with self._acquire_slot():
                response = await self.client.get("/api/tags", timeout=5)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.debug("Health check failed: %s", exc)
            return False
        else:
            return response.status_code == HTTPStatus.OK

    async def get_model_info(self, model: str) -> dict[str, Any] | None:
        models = await self.list_models()
        for item in models:
            if item.get("name") == model:
                return item
        return None


__all__ = ["AsyncSharedOllamaClient", "AsyncOllamaConfig"]

