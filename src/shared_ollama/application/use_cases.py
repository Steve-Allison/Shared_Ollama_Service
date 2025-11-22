"""Use cases for the Shared Ollama Service.

This module defines application use cases that orchestrate domain logic and
coordinate between domain entities and infrastructure adapters. Use cases
contain application-specific business rules but no framework or infrastructure
dependencies.

Design Principles:
    - Dependency Inversion: Depend on interfaces (Protocols), not implementations
    - Single Responsibility: Each use case handles one business operation
    - Orchestration: Coordinate domain logic, logging, metrics, and client calls
    - Framework-agnostic: No FastAPI, Pydantic, or other framework dependencies

Use Case Responsibilities:
    - Validate domain entities (delegated to entities themselves)
    - Coordinate infrastructure adapters (client, logger, metrics)
    - Transform between domain and infrastructure formats
    - Handle streaming vs non-streaming responses
    - Record metrics and log requests
    - Propagate domain exceptions

Key Use Cases:
    - GenerateUseCase: Single-prompt text generation
    - ChatUseCase: Multi-turn text conversations
    - ListModelsUseCase: Retrieve available models
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from shared_ollama.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    GenerationRequest,
    ModelInfo,
    Tool,
)
from shared_ollama.domain.exceptions import InvalidRequestError

if TYPE_CHECKING:
    from shared_ollama.application.interfaces import (
        AnalyticsCollectorInterface,
        MetricsCollectorInterface,
        OllamaClientInterface,
        PerformanceCollectorInterface,
        RequestLoggerInterface,
    )


class GenerateUseCase:
    """Use case for text generation.

    Orchestrates single-prompt text generation requests, handling domain
    validation, client coordination, metrics collection, and logging.

    This use case transforms domain entities (GenerationRequest) into
    infrastructure calls, handles streaming and non-streaming responses,
    and records observability data.

    Attributes:
        _client: Ollama client adapter implementing OllamaClientInterface.
        _logger: Request logger adapter implementing RequestLoggerInterface.
        _metrics: Metrics collector adapter implementing MetricsCollectorInterface.

    Note:
        All dependencies are injected via constructor following dependency
        inversion principle. Use cases depend on interfaces, not concrete
        implementations.
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
        analytics: AnalyticsCollectorInterface | None = None,
        performance: PerformanceCollectorInterface | None = None,
    ) -> None:
        """Initialize the generate use case.

        Args:
            client: Ollama client adapter for making generation requests.
            logger: Request logger for recording request events.
            metrics: Metrics collector for tracking performance and usage.
            analytics: Optional analytics collector for project-based tracking.
            performance: Optional performance collector for detailed metrics.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics
        self._analytics = analytics
        self._performance = performance

    async def execute(
        self,
        request: GenerationRequest,
        request_id: str,
        client_ip: str | None = None,
        project_name: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Execute a generation request.

        Orchestrates the complete generation workflow:
        1. Extracts values from domain entities (prompt, model, options)
        2. Transforms domain format to client format (options dict, tools list)
        3. Calls client adapter to generate text (streaming or non-streaming)
        4. Records metrics and logs request
        5. Returns result in appropriate format

        Args:
            request: Generation request domain entity. Already validated by
                domain layer (GenerationRequest.__post_init__).
            request_id: Unique request identifier for tracing and logging.
            client_ip: Client IP address for logging and analytics. None if
                unavailable.
            project_name: Project name for request tracking and analytics.
                None if not provided.
            stream: Whether to stream the response. True returns AsyncIterator,
                False returns dict.

        Returns:
            If stream=False: dict containing generation result with keys:
                - text: Generated text
                - model: Model name used
                - prompt_eval_count: Prompt tokens evaluated
                - eval_count: Generation tokens produced
                - total_duration: Total generation time (nanoseconds)
                - load_duration: Model load time (nanoseconds)
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with metrics.

        Raises:
            TypeError: If stream=True but result is not AsyncIterator, or
                stream=False but result is not dict.
            InvalidRequestError: If request violates business rules (rare,
                as domain entities validate themselves).
            ConnectionError: If Ollama service is unavailable.
            Exception: For other client or infrastructure errors.

        Side Effects:
            - Records metrics via MetricsCollectorInterface
            - Logs request via RequestLoggerInterface
            - May acquire semaphore/concurrency limits in client adapter
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
                if not isinstance(result, AsyncIterator):
                    raise TypeError("Expected streaming iterator for generate()")
                return result
            if not isinstance(result, dict):
                raise TypeError("Expected dict response for non-streaming generate()")
            result_dict = result
            # Non-streaming: log immediately
            model_used = result_dict.get("model", model_str or "unknown")
            # Extract performance metrics from Ollama response
            load_duration = result_dict.get("load_duration", 0)
            model_load_ms = round(load_duration / 1_000_000, 3) if load_duration else None
            model_warm_start = (load_duration == 0) if load_duration is not None else None

            self._logger.log_request(
                {
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
                }
            )

            self._metrics.record_request(
                model=result_dict.get("model", model_str or "unknown"),
                operation="generate",
                latency_ms=latency_ms,
                success=True,
            )

            # Record project-based analytics (via injected interface)
            if self._analytics:
                self._analytics.record_request_with_project(
                    model=model_used,
                    operation="generate",
                    latency_ms=latency_ms,
                    success=True,
                    project=project_name,
                )

            # Record detailed performance metrics (via injected interface)
            if self._performance:
                self._performance.record_performance(
                    model=model_used,
                    operation="generate",
                    total_latency_ms=latency_ms,
                    success=True,
                    response=result_dict,
                )

            return result_dict

        except ValueError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            model_str = request.model.value if request.model else "unknown"

            self._logger.log_request(
                {
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
                }
            )

            self._metrics.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error="ValueError",
            )

            # Record project-based analytics for errors too (via injected interface)
            if self._analytics:
                self._analytics.record_request_with_project(
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

            self._logger.log_request(
                {
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
                }
            )

            self._metrics.record_request(
                model=model_str,
                operation="generate",
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )

            # Record project-based analytics for errors too (via injected interface)
            if self._analytics:
                self._analytics.record_request_with_project(
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

    Orchestrates multi-turn conversation requests, handling domain validation,
    client coordination, metrics collection, and logging.

    This use case transforms domain entities (ChatRequest) into infrastructure
    calls, handles streaming and non-streaming responses, and records
    observability data for conversation-based interactions.

    Attributes:
        _client: Ollama client adapter implementing OllamaClientInterface.
        _logger: Request logger adapter implementing RequestLoggerInterface.
        _metrics: Metrics collector adapter implementing MetricsCollectorInterface.

    Note:
        ChatUseCase handles multi-turn conversations with message history,
        unlike GenerateUseCase which handles single-prompt generation.
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
        analytics: AnalyticsCollectorInterface | None = None,
        performance: PerformanceCollectorInterface | None = None,
    ) -> None:
        """Initialize the chat use case.

        Args:
            client: Ollama client adapter for making chat requests.
            logger: Request logger for recording request events.
            metrics: Metrics collector for tracking performance and usage.
            analytics: Optional analytics collector for project-based tracking.
            performance: Optional performance collector for detailed metrics.
        """
        self._client = client
        self._logger = logger
        self._metrics = metrics
        self._analytics = analytics
        self._performance = performance

    def _convert_message_to_dict(self, msg: ChatMessage) -> dict[str, Any]:
        """Convert a domain ChatMessage to client format dict."""
        message_dict: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            message_dict["content"] = msg.content
        if msg.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            message_dict["tool_call_id"] = msg.tool_call_id
        return message_dict

    def _convert_options_to_dict(self, options: GenerationOptions | None) -> dict[str, Any] | None:
        """Convert domain GenerationOptions to client format dict."""
        if not options:
            return None
        options_dict = {
            "temperature": options.temperature,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "repeat_penalty": options.repeat_penalty,
            "num_predict": options.max_tokens,
            "seed": options.seed,
            "stop": options.stop,
        }
        return {k: v for k, v in options_dict.items() if v is not None}

    def _convert_tools_to_list(self, tools: tuple[Tool, ...] | None) -> list[dict[str, Any]] | None:
        """Convert domain Tools to client format list."""
        if not tools:
            return None
        return [
            {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                },
            }
            for tool in tools
        ]

    def _record_success(
        self,
        model: str,
        operation: str,
        latency_ms: float,
        request_id: str,
        client_ip: str | None,
        project_name: str | None,
        result_dict: dict[str, Any],
    ) -> None:
        """Record success metrics, logs, and analytics."""
        load_duration = result_dict.get("load_duration", 0)
        model_load_ms = round(load_duration / 1_000_000, 3) if load_duration else None
        model_warm_start = (load_duration == 0) if load_duration is not None else None

        self._logger.log_request(
            {
                "event": "api_request",
                "client_type": "rest_api",
                "operation": operation,
                "status": "success",
                "model": model,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "model_load_ms": model_load_ms,
                "model_warm_start": model_warm_start,
            }
        )
        self._metrics.record_request(
            model=model, operation=operation, latency_ms=latency_ms, success=True
        )
        if self._analytics:
            self._analytics.record_request_with_project(
                model=model,
                operation=operation,
                latency_ms=latency_ms,
                success=True,
                project=project_name,
            )
        if self._performance:
            self._performance.record_performance(
                model=model,
                operation=operation,
                total_latency_ms=latency_ms,
                success=True,
                response=result_dict,
            )

    def _record_error(
        self,
        model: str,
        operation: str,
        latency_ms: float,
        request_id: str,
        client_ip: str | None,
        project_name: str | None,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record error metrics, logs, and analytics."""
        self._logger.log_request(
            {
                "event": "api_request",
                "client_type": "rest_api",
                "operation": operation,
                "status": "error",
                "model": model,
                "request_id": request_id,
                "client_ip": client_ip,
                "project_name": project_name,
                "latency_ms": round(latency_ms, 3),
                "error_type": error_type,
                "error_message": error_message,
            }
        )
        self._metrics.record_request(
            model=model, operation=operation, latency_ms=latency_ms, success=False, error=error_type
        )
        if self._analytics:
            self._analytics.record_request_with_project(
                model=model,
                operation=operation,
                latency_ms=latency_ms,
                success=False,
                project=project_name,
                error=error_type,
            )

    async def execute(
        self,
        request: ChatRequest,
        request_id: str,
        client_ip: str | None = None,
        project_name: str | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Execute a chat completion request.

        Orchestrates the complete chat workflow:
        1. Transforms domain messages to client format (with tool calling support)
        2. Extracts model, options, format, and tools from domain entities
        3. Calls client adapter to complete chat (streaming or non-streaming)
        4. Records metrics and logs request
        5. Returns result in appropriate format

        Args:
            request: Chat request domain entity. Already validated by domain
                layer (ChatRequest.__post_init__). Contains message history.
            request_id: Unique request identifier for tracing and logging.
            client_ip: Client IP address for logging and analytics. None if
                unavailable.
            project_name: Project name for request tracking and analytics.
                None if not provided.
            stream: Whether to stream the response. True returns AsyncIterator,
                False returns dict.

        Returns:
            If stream=False: dict containing chat result with keys:
                - message: Assistant message dict with role and content
                - model: Model name used
                - prompt_eval_count: Prompt tokens evaluated
                - eval_count: Generation tokens produced
                - total_duration: Total generation time (nanoseconds)
                - load_duration: Model load time (nanoseconds)
            If stream=True: AsyncIterator[dict[str, Any]] yielding chunks
                with incremental text and final chunk with metrics.

        Raises:
            TypeError: If stream=True but result is not AsyncIterator, or
                stream=False but result is not dict.
            InvalidRequestError: If request violates business rules (rare,
                as domain entities validate themselves).
            ConnectionError: If Ollama service is unavailable.
            Exception: For other client or infrastructure errors.

        Side Effects:
            - Records metrics via MetricsCollectorInterface
            - Logs request via RequestLoggerInterface
            - May acquire semaphore/concurrency limits in client adapter
        """
        start_time = time.perf_counter()
        model_str = request.model.value if request.model else None

        try:
            # Convert domain entities to client format using helper methods
            messages = [self._convert_message_to_dict(msg) for msg in request.messages]
            options_dict = self._convert_options_to_dict(request.options)
            tools_list = self._convert_tools_to_list(request.tools)

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

            # Handle streaming response (metrics logged by caller)
            if stream:
                if not isinstance(result, AsyncIterator):
                    raise TypeError("Expected streaming iterator for chat()")
                return result

            # Handle non-streaming response
            if not isinstance(result, dict):
                raise TypeError("Expected dict response for non-streaming chat()")

            model_used = result.get("model", model_str or "unknown")
            self._record_success(
                model_used, "chat", latency_ms, request_id, client_ip, project_name, result
            )
            return result

        except ValueError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._record_error(
                model_str or "unknown",
                "chat",
                latency_ms,
                request_id,
                client_ip,
                project_name,
                "ValueError",
                str(exc),
            )
            raise InvalidRequestError(f"Invalid request: {exc!s}") from exc

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._record_error(
                model_str or "unknown",
                "chat",
                latency_ms,
                request_id,
                client_ip,
                project_name,
                type(exc).__name__,
                str(exc),
            )
            raise


class ListModelsUseCase:
    """Use case for listing available models.

    Orchestrates model listing requests, handling client coordination,
    domain entity conversion, metrics collection, and logging.

    This use case retrieves available models from the Ollama service and
    converts them to domain entities (ModelInfo) for use by the application.

    Attributes:
        _client: Ollama client adapter implementing OllamaClientInterface.
        _logger: Request logger adapter implementing RequestLoggerInterface.
        _metrics: Metrics collector adapter implementing MetricsCollectorInterface.

    Note:
        This use case doesn't require a domain request entity since listing
        models has no input parameters beyond request metadata.
    """

    def __init__(
        self,
        client: OllamaClientInterface,
        logger: RequestLoggerInterface,
        metrics: MetricsCollectorInterface,
    ) -> None:
        """Initialize the list models use case.

        Args:
            client: Ollama client adapter for listing models.
            logger: Request logger for recording request events.
            metrics: Metrics collector for tracking performance and usage.
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

        Orchestrates the complete list models workflow:
        1. Calls client adapter to retrieve available models
        2. Converts infrastructure format (dicts) to domain entities (ModelInfo)
        3. Records metrics and logs request
        4. Returns list of domain entities

        Args:
            request_id: Unique request identifier for tracing and logging.
            client_ip: Client IP address for logging and analytics. None if
                unavailable.
            project_name: Project name for request tracking and analytics.
                None if not provided.

        Returns:
            List of ModelInfo domain entities representing available models.
            Each ModelInfo contains name, optional size, and optional modified_at.

        Raises:
            ConnectionError: If Ollama service is unavailable.
            Exception: For other client or infrastructure errors.

        Side Effects:
            - Records metrics via MetricsCollectorInterface
            - Logs request via RequestLoggerInterface
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
            self._logger.log_request(
                {
                    "event": "api_request",
                    "client_type": "rest_api",
                    "operation": "list_models",
                    "status": "success",
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "project_name": project_name,
                    "latency_ms": round(latency_ms, 3),
                }
            )

            self._metrics.record_request(
                model="system",
                operation="list_models",
                latency_ms=latency_ms,
                success=True,
            )

            return models

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._logger.log_request(
                {
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
                }
            )

            self._metrics.record_request(
                model="system",
                operation="list_models",
                latency_ms=latency_ms,
                success=False,
                error=type(exc).__name__,
            )

            raise
