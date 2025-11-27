"""System routes for health, models, queue stats, and observability.

Provides endpoints for system monitoring, health checks, and observability.
These endpoints enable external systems to monitor service health, query
available models, and access metrics and analytics.

Key Features:
    - Health Checks: Service availability and Ollama connectivity
    - Model Management: List available models with metadata
    - Queue Statistics: Request queue status and metrics
    - Performance Metrics: Detailed performance statistics
    - Analytics: Project-based usage analytics
    - Rate Limiting: Integrated with slowapi rate limiting

Endpoints:
    GET /api/v1/health
        - Response: HealthResponse (service status, Ollama connectivity)
        - Rate Limited: No (health checks should be fast)

    GET /api/v1/models
        - Response: ModelsResponse (list of available models with metadata)
        - Rate Limited: Yes (via slowapi middleware)

    GET /api/v1/queue/stats
        - Response: QueueStatsResponse (chat and VLM queue statistics)
        - Rate Limited: Yes (via slowapi middleware)

    GET /api/v1/metrics
        - Response: MetricsResponse (service metrics with time window)
        - Query Params: window_minutes (optional, default: all time)
        - Rate Limited: Yes (via slowapi middleware)

    GET /api/v1/performance/stats
        - Response: PerformanceStatsResponse (detailed performance metrics)
        - Query Params: window_minutes (optional, default: all time)
        - Rate Limited: Yes (via slowapi middleware)

    GET /api/v1/analytics
        - Response: AnalyticsResponse (project-based analytics)
        - Query Params: window_minutes, project (optional filters)
        - Rate Limited: Yes (via slowapi middleware)

Observability:
    - All endpoints provide structured data for monitoring
    - Time window filtering for metrics and analytics
    - Project-based filtering for analytics
    - Comprehensive statistics for performance analysis
"""

from __future__ import annotations

import copy
import functools
import json
import logging
from datetime import UTC, datetime
from importlib import resources
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

from shared_ollama.api.dependencies import (
    get_chat_queue,
    get_list_models_use_case,
    get_request_context,
)
from shared_ollama.api.error_handlers import _map_http_status_code
from shared_ollama.api.mappers import domain_to_api_model_info
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import (
    AnalyticsResponse,
    HealthResponse,
    MetricsResponse,
    ModelProfileResponse,
    ModelsResponse,
    PerformanceStatsResponse,
    QueueStatsResponse,
)
from shared_ollama.application.use_cases import ListModelsUseCase
from shared_ollama.core.queue import RequestQueue
from shared_ollama.core.utils import check_service_health, get_model_profile_summary

logger = logging.getLogger(__name__)

router = APIRouter()

QueueDep = Annotated[RequestQueue, Depends(get_chat_queue)]
ListModelsDep = Annotated[ListModelsUseCase, Depends(get_list_models_use_case)]

_PDL_TEMPLATE_PACKAGE = "shared_ollama.data.pdl"
_PDL_TEMPLATE_FILES = {
    "vlm": "vlm_insights_template.json",
    "chat": "text_chat_template.json",
}


@functools.cache
def _load_pdl_template(filename: str) -> dict[str, Any]:
    template_path = resources.files(_PDL_TEMPLATE_PACKAGE).joinpath(filename)
    with template_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _prepare_pdl_template(filename: str, request: Request) -> dict[str, Any]:
    template = copy.deepcopy(_load_pdl_template(filename))
    base_url = str(request.base_url).rstrip("/")

    runtime_section = template.get("runtime", {})
    invoke_section = runtime_section.get("invoke")
    if isinstance(invoke_section, dict):
        url = invoke_section.get("url")
        if isinstance(url, str):
            invoke_section["url"] = url.replace("__API_BASE_URL__", base_url)

    metadata = template.setdefault("metadata", {})
    metadata["served_at"] = datetime.now(UTC).isoformat()
    metadata["base_url"] = base_url
    metadata["source"] = "shared-ollama.api.system"

    return template


@router.get("/pdl/templates", tags=["PDL"])
async def list_pdl_templates() -> dict[str, list[str]]:
    """List the IBM PDL templates exposed by this API."""

    return {"templates": sorted(_PDL_TEMPLATE_FILES.keys())}


@router.get("/pdl/templates/{template_name}", tags=["PDL"])
async def get_pdl_template(template_name: str, request: Request) -> dict[str, Any]:
    """Return a concrete IBM PDL template with runtime details injected."""

    filename = _PDL_TEMPLATE_FILES.get(template_name)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Unknown PDL template '{template_name}'. "
                f"Available templates: {', '.join(sorted(_PDL_TEMPLATE_FILES))}"
            ),
        )

    return _prepare_pdl_template(filename, request)


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Performs a lightweight health check on the API and underlying Ollama
    service. Returns status information without requiring authentication.

    Returns:
        HealthResponse with:
            - status: "healthy" or "unhealthy"
            - ollama_service: Ollama service status string
            - version: API version string

    Side effects:
        Calls check_service_health() which makes HTTP request to Ollama.
    """
    is_healthy, error = check_service_health()
    ollama_status = "healthy" if is_healthy else f"unhealthy: {error}"

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        ollama_service=ollama_status,
        version="2.0.0",
    )


@router.get("/queue/stats", response_model=QueueStatsResponse, tags=["Queue"])
async def get_queue_stats(
    queue: RequestQueue = Depends(get_chat_queue),  # noqa: B008
) -> QueueStatsResponse:
    """Get chat queue statistics.

    Returns comprehensive queue metrics for the chat queue including current state,
    historical counts, and performance statistics.

    Args:
        queue: Chat request queue (injected via dependency injection).

    Returns:
        QueueStatsResponse with:
            - Current state: queued, in_progress
            - Historical counts: completed, failed, rejected, timeout
            - Performance metrics: wait times (total, max, avg)
            - Configuration: max_concurrent, max_queue_size, default_timeout

    Side effects:
        Acquires asyncio.Lock briefly to read statistics atomically.
    """
    stats = await queue.get_stats()
    config = queue.get_config()

    return QueueStatsResponse(
        queued=stats.queued,
        in_progress=stats.in_progress,
        completed=stats.completed,
        failed=stats.failed,
        rejected=stats.rejected,
        timeout=stats.timeout,
        total_wait_time_ms=stats.total_wait_time_ms,
        max_wait_time_ms=stats.max_wait_time_ms,
        avg_wait_time_ms=stats.avg_wait_time_ms,
        max_concurrent=config["max_concurrent"],
        max_queue_size=config["max_queue_size"],
        default_timeout=config["default_timeout"],
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(
    window_minutes: int | None = None,
) -> MetricsResponse:
    """Get service metrics.

    Returns comprehensive service metrics including request counts, latency
    statistics, and error breakdowns. Supports optional time-window filtering.

    Args:
        window_minutes: Optional time window in minutes. If provided, only
            metrics from the last N minutes are included. If None, returns
            all metrics since service startup.

    Returns:
        Dictionary with metrics including:
            - total_requests: Total number of requests
            - successful_requests: Number of successful requests
            - failed_requests: Number of failed requests
            - requests_by_model: Request counts by model
            - requests_by_operation: Request counts by operation
            - average_latency_ms: Average request latency
            - p50_latency_ms: 50th percentile (median) latency
            - p95_latency_ms: 95th percentile latency
            - p99_latency_ms: 99th percentile latency
            - errors_by_type: Error counts by error type
            - last_request_time: Timestamp of most recent request
            - first_request_time: Timestamp of oldest request
    """
    from shared_ollama.telemetry.metrics import get_metrics_endpoint

    data = get_metrics_endpoint()
    return MetricsResponse.model_validate(data)


@router.get("/performance/stats", response_model=PerformanceStatsResponse, tags=["Performance"])
async def get_performance_stats() -> PerformanceStatsResponse:
    """Get aggregated performance statistics.

    Returns detailed performance metrics including token generation rates,
    model load times, and generation times. Statistics are grouped by model.

    Returns:
        Dictionary with performance statistics including:
            - avg_tokens_per_second: Overall average token generation rate
            - avg_load_time_ms: Average model load time
            - avg_generation_time_ms: Average generation time
            - total_requests: Count of successful requests with performance data
            - by_model: Per-model statistics with same structure plus request_count

        Returns empty dict if no performance data is available.
    """
    from shared_ollama.telemetry.performance import get_performance_stats as _get_performance_stats

    data = _get_performance_stats()
    if not data:
        data = {
            "avg_tokens_per_second": 0.0,
            "avg_load_time_ms": 0.0,
            "avg_generation_time_ms": 0.0,
            "total_requests": 0,
            "by_model": {},
        }
    return PerformanceStatsResponse.model_validate(data)


@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics(
    window_minutes: int | None = None,
    project: str | None = None,
) -> AnalyticsResponse:
    """Get comprehensive analytics report.

    Returns project-based analytics with time-series data, latency percentiles,
    and usage breakdowns. Supports filtering by time window and project.

    Args:
        window_minutes: Optional time window in minutes. If provided, only
            analytics from the last N minutes are included. If None, returns
            all analytics since service startup.
        project: Optional project name filter. If provided, only analytics
            for the specified project are returned. If None, returns analytics
            for all projects.

    Returns:
        Dictionary with analytics including:
            - total_requests: Total number of requests
            - successful_requests: Number of successful requests
            - failed_requests: Number of failed requests
            - success_rate: Success rate (0.0 to 1.0)
            - average_latency_ms: Average request latency
            - p50_latency_ms, p95_latency_ms, p99_latency_ms: Latency percentiles
            - requests_by_model: Request counts by model
            - requests_by_operation: Request counts by operation
            - requests_by_project: Request counts by project
            - project_metrics: Detailed metrics per project
            - hourly_metrics: Time-series data (hourly aggregation)
            - start_time, end_time: Time range of data
    """
    from shared_ollama.telemetry.analytics import get_analytics_json

    data = get_analytics_json(window_minutes=window_minutes, project=project)
    return AnalyticsResponse.model_validate(data)


@router.get("/models", response_model=ModelsResponse, tags=["Models"])
@limiter.limit("30/minute")
async def list_models(
    request: Request,
    use_case: ListModelsUseCase = Depends(get_list_models_use_case),  # noqa: B008
) -> ModelsResponse:
    """List available models.

    Retrieves the list of all models available in the Ollama service.
    Rate limited to 30 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected).
        use_case: ListModelsUseCase instance (injected via dependency injection).

    Returns:
        ModelsResponse with list of ModelInfo objects, one for each available model.

    Raises:
        HTTPException: If Ollama service is unavailable or request fails.
    """
    ctx = get_request_context(request)

    try:
        # Use case handles all business logic, logging, and metrics
        domain_models = await use_case.execute(
            request_id=ctx.request_id,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
        )

        # Convert domain entities to API models
        api_models = [domain_to_api_model_info(model) for model in domain_models]

        return ModelsResponse(models=api_models)

    except ConnectionError as exc:
        logger.error("connection_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is unavailable. Please check if the service is running.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response else None
        http_status, error_msg = _map_http_status_code(status_code)
        logger.error(
            "http_status_error: request_id=%s, status_code=%s, error=%s",
            ctx.request_id,
            status_code,
            str(exc),
        )
        raise HTTPException(status_code=http_status, detail=error_msg) from exc
    except httpx.RequestError as exc:
        logger.error("request_error: request_id=%s, error=%s", ctx.request_id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to Ollama service. Please check if the service is running.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_listing_models: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        # Include request_id in error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


@router.get("/system/model-profile", response_model=ModelProfileResponse, tags=["Models"])
async def get_model_profile() -> ModelProfileResponse:
    """Return the active hardware profile and model recommendations."""

    summary = get_model_profile_summary()
    return ModelProfileResponse(**summary)
