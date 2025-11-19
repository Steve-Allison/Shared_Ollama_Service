"""Use cases for the Shared Ollama Service.

Use cases orchestrate domain logic and coordinate between domain entities
and infrastructure adapters. They contain application-specific business rules
but no framework or infrastructure dependencies.

All use cases depend only on:
    - Domain entities and value objects
    - Interface protocols (not implementations)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from shared_ollama.domain.entities import (
    ChatRequest,
    GenerationRequest,
    ModelInfo,
)
from shared_ollama.domain.exceptions import InvalidRequestError

if TYPE_CHECKING:
    from shared_ollama.application.interfaces import (
        MetricsCollectorInterface,
        OllamaClientInterface,
        RequestLoggerInterface,
    )


class GenerateUseCase:
    """Use case for text generation.

    Orchestrates text generation requests, handling domain validation,
    client coordination, and metrics/logging.

    Attributes:
        client: Ollama client implementation (satisfies OllamaClientInterface).
        logger: Request logger implementation (satisfies RequestLoggerInterface).
        metrics: Metrics collector implementation (satisfies MetricsCollectorInterface).
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
    ) -> None:
        """Initialize the generate use case.

        Args:
            client: Ollama client implementation.
            logger: Request logger implementation.
            metrics: Metrics collector implementation.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics

    async def execute(
        self,
        request: GenerationRequest,
        request_id: str,
        client_ip: str | None = None,
        project_name: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Execute a generation request.

        Orchestrates the generation workflow:
        1. Validates request (domain validation)
        2. Calls client to generate text
        3. Logs request and records metrics
        4. Returns result

        Args:
            request: Generation request domain entity.
            request_id: Unique request identifier.
            client_ip: Client IP address for logging.
            project_name: Project name for logging.
            stream: Whether to stream the response.

        Returns:
            - dict with generation result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            InvalidRequestError: If request violates business rules.
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        start_time = time.perf_counter()

        try:
            # Convert domain entities to client format
            prompt_str = request.prompt.value
            model_str = request.model.value if request.model else None
            system_str = request.system.value if request.system else None

            # Convert options to dict format
            options_dict: dict[str, Any] | None = None
            if request.options:
                options_dict = {
                    "temperature": request.options.temperature,
                    "top_p": request.options.top_p,
                    "top_k": request.options.top_k,
                    "repeat_penalty": request.options.repeat_penalty,
                    "num_predict": request.options.max_tokens,
                    "seed": request.options.seed,
                    "stop": request.options.stop,
                }
                # Remove None values
                options_dict = {k: v for k, v in options_dict.items() if v is not None}

            # Convert tools to dict format (POML compatible)
            tools_list: list[dict[str, Any]] | None = None
            if request.tools:
                tools_list = [
                    {
                        "type": tool.type,
                        "function": {
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": tool.function.parameters,
                        },
                    }
                    for tool in request.tools
                ]

            # Call client with format and tools support
            result = await self._client.generate(
                prompt=prompt_str,
                model=model_str,
                system=system_str,
                options=options_dict,
                stream=stream,
                format=request.format,
                tools=tools_list,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log and record metrics
            if stream:
                # For streaming, we'll log after first chunk
                return result
            # Non-streaming: log immediately
            model_used = result.get("model", model_str or "unknown")
            # Extract performance metrics from Ollama response
            load_duration = result.get("load_duration", 0)
            model_load_ms = round(load_duration / 1_000_000, 3) if load_duration else None
            model_warm_start = (load_duration == 0) if load_duration is not None else None

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "generate",
                "status": "success",
                "model": model_used,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "model_load_ms": model_load_ms,
                "model_warm_start": model_warm_start,
            })

            self._metrics.record_request(
                model=result.get("model", model_str or "unknown"),
                operation="generate",
                latency_ms=latency_ms,
                success=True,
            )

            # Record project-based analytics
            from shared_ollama.telemetry.analytics import AnalyticsCollector
            AnalyticsCollector.record_request_with_project(
                model=model_used,
                operation="generate",
                latency_ms=latency_ms,
                success=True,
                project=project_name,
            )

            # Record detailed performance metrics
            from shared_ollama.telemetry.performance import PerformanceCollector
            PerformanceCollector.record_performance(
                model=model_used,
                operation="generate",
                total_latency_ms=latency_ms,
                success=True,
                response=result,  # Pass dict - PerformanceCollector now handles it
            )

            return result

        except ValueError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "generate",
                "status": "error",
                "model": model_str,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "error_type": "ValueError",
                "error_message": str(exc),
            })

            self._metrics.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error="ValueError",
            )

            # Record project-based analytics for errors too
            from shared_ollama.telemetry.analytics import AnalyticsCollector
            AnalyticsCollector.record_request_with_project(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                project=project_name,
                error="ValueError",
            )

            raise InvalidRequestError(f"Invalid request: {exc!s}") from exc
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "generate",
                "status": "error",
                "model": model_str,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

            self._metrics.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )

            # Record project-based analytics for errors too
            from shared_ollama.telemetry.analytics import AnalyticsCollector
            AnalyticsCollector.record_request_with_project(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                project=project_name,
                error=type(exc).__name__,
            )

            raise


class ChatUseCase:
    """Use case for chat completion.

    Orchestrates chat completion requests, handling domain validation,
    client coordination, and metrics/logging.

    Attributes:
        client: Ollama client implementation (satisfies OllamaClientInterface).
        logger: Request logger implementation (satisfies RequestLoggerInterface).
        metrics: Metrics collector implementation (satisfies MetricsCollectorInterface).
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
    ) -> None:
        """Initialize the chat use case.

        Args:
            client: Ollama client implementation.
            logger: Request logger implementation.
            metrics: Metrics collector implementation.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics

    async def execute(
        self,
        request: ChatRequest,
        request_id: str,
        client_ip: str | None = None,
        project_name: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Execute a chat completion request.

        Orchestrates the chat workflow:
        1. Validates request (domain validation)
        2. Calls client to complete chat
        3. Logs request and records metrics
        4. Returns result

        Args:
            request: Chat request domain entity.
            request_id: Unique request identifier.
            client_ip: Client IP address for logging.
            project_name: Project name for logging.
            stream: Whether to stream the response.

        Returns:
            - dict with chat result if stream=False
            - AsyncIterator of chunks if stream=True

        Raises:
            InvalidRequestError: If request violates business rules.
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        start_time = time.perf_counter()

        try:
            # Convert domain entities to client format with tool calling support
            messages: list[dict[str, Any]] = []

            for msg in request.messages:
                # Build message dict with tool calling support
                message_dict: dict[str, Any] = {"role": msg.role}

                # Add content if present
                if msg.content is not None:
                    message_dict["content"] = msg.content

                # Add tool_calls if present
                if msg.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                # Add tool_call_id if present (for tool response messages)
                if msg.tool_call_id:
                    message_dict["tool_call_id"] = msg.tool_call_id

                messages.append(message_dict)

            model_str = request.model.value if request.model else None

            # Convert options to dict format
            options_dict: dict[str, Any] | None = None
            if request.options:
                options_dict = {
                    "temperature": request.options.temperature,
                    "top_p": request.options.top_p,
                    "top_k": request.options.top_k,
                    "repeat_penalty": request.options.repeat_penalty,
                    "num_predict": request.options.max_tokens,
                    "seed": request.options.seed,
                    "stop": request.options.stop,
                }
                # Remove None values
                options_dict = {k: v for k, v in options_dict.items() if v is not None}

            # Convert tools to dict format (POML compatible)
            tools_list: list[dict[str, Any]] | None = None
            if request.tools:
                tools_list = [
                    {
                        "type": tool.type,
                        "function": {
                            "name": tool.function.name,
                            "description": tool.function.description,
                            "parameters": tool.function.parameters,
                        },
                    }
                    for tool in request.tools
                ]

            # Call client with format and tools support
            result = await self._client.chat(
                messages=messages,
                model=model_str,
                options=options_dict,
                stream=stream,
                format=request.format,
                tools=tools_list,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log and record metrics
            if stream:
                # For streaming, we'll log after first chunk
                return result
            # Non-streaming: log immediately
            model_used = result.get("model", model_str or "unknown")
            # Extract performance metrics from Ollama response
            load_duration = result.get("load_duration", 0)
            model_load_ms = round(load_duration / 1_000_000, 3) if load_duration else None
            model_warm_start = (load_duration == 0) if load_duration is not None else None

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "chat",
                "status": "success",
                "model": model_used,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "model_load_ms": model_load_ms,
                "model_warm_start": model_warm_start,
            })

            self._metrics.record_request(
                model=model_used,
                operation="chat",
                latency_ms=latency_ms,
                success=True,
            )

            # Record project-based analytics
            from shared_ollama.telemetry.analytics import AnalyticsCollector
            AnalyticsCollector.record_request_with_project(
                model=model_used,
                operation="chat",
                latency_ms=latency_ms,
                success=True,
                project=project_name,
            )

            # Record detailed performance metrics
            from shared_ollama.telemetry.performance import PerformanceCollector
            PerformanceCollector.record_performance(
                model=model_used,
                operation="chat",
                total_latency_ms=latency_ms,
                success=True,
                response=result,  # Pass dict - PerformanceCollector now handles it
            )

            return result

        except ValueError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "chat",
                "status": "error",
                "model": model_str,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "error_type": "ValueError",
                "error_message": str(exc),
            })

            self._metrics.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                error="ValueError",
            )

            # Record project-based analytics for errors too
            from shared_ollama.telemetry.analytics import AnalyticsCollector
            AnalyticsCollector.record_request_with_project(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                project=project_name,
                error="ValueError",
            )

            raise InvalidRequestError(f"Invalid request: {exc!s}") from exc
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "chat",
                "status": "error",
                "model": model_str,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

            self._metrics.record_request(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )

            # Record project-based analytics for errors too
            from shared_ollama.telemetry.analytics import AnalyticsCollector
            AnalyticsCollector.record_request_with_project(
                model=model_str,
                operation="chat",
                latency_ms=latency_ms,
                success=False,
                project=project_name,
                error=type(exc).__name__,
            )

            raise


class ListModelsUseCase:
    """Use case for listing available models.

    Orchestrates model listing, handling client coordination and
    metrics/logging.

    Attributes:
        client: Ollama client implementation (satisfies OllamaClientInterface).
        logger: Request logger implementation (satisfies RequestLoggerInterface).
        metrics: Metrics collector implementation (satisfies MetricsCollectorInterface).
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
    ) -> None:
        """Initialize the list models use case.

        Args:
            client: Ollama client implementation.
            logger: Request logger implementation.
            metrics: Metrics collector implementation.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics

    async def execute(
        self,
        request_id: str,
        client_ip: str | None = None,
        project_name: str | None = None,
    ) -> list[ModelInfo]:
        """Execute a list models request.

        Orchestrates the list models workflow:
        1. Calls client to list models
        2. Converts to domain entities
        3. Logs request and records metrics
        4. Returns domain entities

        Returns:
            List of ModelInfo domain entities.

        Raises:
            ConnectionError: If service is unavailable.
            Exception: For other errors.
        """
        start_time = time.perf_counter()

        try:
            models_data = await self._client.list_models()

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Convert to domain entities
            models = [
                ModelInfo(
                    name=model.get("name", "unknown"),
                    size=model.get("size"),
                    modified_at=model.get("modified_at"),
                )
                for model in models_data
            ]

            # Log and record metrics
            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "list_models",
                "status": "success",
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
            })

            self._metrics.record_request(
                model="system",
                operation="list_models",
                latency_ms=latency_ms,
                success=True,
            )

            return models

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._logger.log_request({
                "event": "api_request",
                "client_type": "rest_api",
                "operation": "list_models",
                "status": "error",
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

            self._metrics.record_request(
                model="system",
                operation="list_models",
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )

            raise
