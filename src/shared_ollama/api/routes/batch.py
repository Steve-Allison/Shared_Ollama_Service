"""Batch routes for batch processing endpoints.

Provides batch endpoints for processing multiple chat and VLM requests
in parallel. Batch processing enables efficient handling of multiple
requests with configurable concurrency limits.

Key Features:
    - Parallel Processing: Multiple requests processed concurrently
    - Concurrency Control: Semaphore-based limiting (configurable)
    - Request Limits: Maximum requests per batch (configurable)
    - Error Isolation: Individual request failures don't fail entire batch
    - Rate Limiting: Integrated with slowapi rate limiting

Endpoints:
    POST /api/v1/batch/chat
        - Request: BatchChatRequest (list of ChatRequest)
        - Response: BatchResponse (list of results with success/error per request)
        - Max Requests: 50 (configurable via BATCH_CHAT_MAX_REQUESTS)
        - Rate Limited: Yes (10/minute via slowapi)

    POST /api/v1/batch/vlm
        - Request: BatchVLMRequest (list of VLMRequest)
        - Response: BatchResponse (list of results with success/error per request)
        - Max Requests: 20 (configurable via BATCH_VLM_MAX_REQUESTS)
        - Rate Limited: Yes (10/minute via slowapi)

Batch Processing:
    - Requests processed concurrently (up to max_concurrent limit)
    - Individual request failures isolated (reported in results)
    - Results include success/error status for each request
    - Total processing time tracked and reported

Request Flow:
    1. Request validated by Pydantic (BatchChatRequest or BatchVLMRequest)
    2. Request count validated (must not exceed max_requests)
    3. Individual requests mapped to domain entities
    4. Executed via BatchChatUseCase or BatchVLMUseCase
    5. Results aggregated into BatchResponse
    6. Returned as JSON with per-request results
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from shared_ollama.api.dependencies import (
    get_batch_chat_use_case,
    get_batch_vlm_use_case,
    get_request_context,
)
from shared_ollama.api.mappers import (
    api_to_domain_chat_request,
    api_to_domain_chat_request_openai,
    api_to_domain_vlm_request,
    api_to_domain_vlm_request_openai,
)
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import (
    BatchChatRequest,
    BatchChatRequestOpenAI,
    BatchResponse,
    BatchVLMRequest,
    BatchVLMRequestOpenAI,
)
from shared_ollama.application.batch_use_cases import (
    BatchChatUseCase,
    BatchVLMUseCase,
)
from shared_ollama.core.utils import get_allowed_models, is_model_allowed
from shared_ollama.domain.exceptions import InvalidRequestError
from shared_ollama.infrastructure.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

BatchChatDep = Annotated[BatchChatUseCase, Depends(get_batch_chat_use_case)]
BatchVLMDep = Annotated[BatchVLMUseCase, Depends(get_batch_vlm_use_case)]


@router.post("/batch/chat", tags=["Batch"], response_model=BatchResponse)
@limiter.limit("10/minute")
async def batch_chat(
    request: Request,
    use_case: BatchChatUseCase = Depends(get_batch_chat_use_case),  # noqa: B008
) -> BatchResponse:
    """Batch text-only chat completion endpoint.

    Processes multiple chat requests in parallel, returning all results in a single
    response. Rate limited to 10 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            BatchChatRequest JSON with list of chat requests (max 50).
        use_case: BatchChatUseCase instance (injected via DI).

    Returns:
        BatchResponse with:
            - batch_id: Unique batch identifier
            - total_requests: Number of requests in batch
            - successful: Count of successful requests
            - failed: Count of failed requests
            - total_time_ms: Total batch processing time
            - results: List of individual results with success/error status

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Too many requests in batch (>50)
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Processes up to 5 requests concurrently
        - Logs all individual requests
        - Records batch-level metrics
    """
    ctx = get_request_context(request)

    # Parse request body
    try:
        body = await request.json()
        api_req = BatchChatRequest(**body)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in request body: {e!s}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e!s}",
        ) from e

    try:
        # Validate batch size
        if len(api_req.requests) > settings.batch.chat_max_requests:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many requests in batch. Maximum allowed: {settings.batch.chat_max_requests}",
            )

        # Validate all models are allowed for current hardware profile
        allowed = get_allowed_models()
        invalid_models = []
        for req in api_req.requests:
            if req.model and not is_model_allowed(req.model):
                invalid_models.append(req.model)
        if invalid_models:
            unique_invalid = sorted(set(invalid_models))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Models not supported on this hardware profile: {', '.join(unique_invalid)}. "
                    f"Allowed models: {', '.join(sorted(allowed))}"
                ),
            )

        # Convert API requests to domain entities
        domain_requests = [api_to_domain_chat_request(req) for req in api_req.requests]

        # Execute batch
        logger.info(
            "batch_chat_requested: request_id=%s, count=%d", ctx.request_id, len(domain_requests)
        )
        result = await use_case.execute(
            requests=domain_requests,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
        )

        return BatchResponse(**result)

    except HTTPException:
        # Re-raise HTTPException
        raise
    except (InvalidRequestError, ValueError) as exc:
        logger.warning(
            "batch_chat_validation_error: request_id=%s, error=%s", ctx.request_id, str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid batch chat request: {exc!s}",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_batch_chat: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc


@router.post("/batch/vlm", tags=["Batch"], response_model=BatchResponse)
@limiter.limit("5/minute")
async def batch_vlm(
    request: Request,
    use_case: BatchVLMUseCase = Depends(get_batch_vlm_use_case),  # noqa: B008
) -> BatchResponse:
    """Batch VLM chat completion endpoint.

    Processes multiple VLM requests in parallel, with image compression and caching.
    Rate limited to 5 requests per minute per IP address due to resource intensity.

    Args:
        request: FastAPI Request object (injected). Body must contain
            BatchVLMRequest JSON with list of VLM requests (max 20).
        use_case: BatchVLMUseCase instance (injected via DI).

    Returns:
        BatchResponse with:
            - batch_id: Unique batch identifier
            - total_requests: Number of requests in batch
            - successful: Count of successful requests
            - failed: Count of failed requests
            - total_time_ms: Total batch processing time
            - results: List of individual results with success/error status

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Too many requests in batch (>20) or invalid images
            - 500: Internal server error

    Side effects:
        - Parses request body JSON
        - Processes up to 3 VLM requests concurrently (resource-intensive)
        - Compresses and caches images
        - Logs all individual requests with VLM metrics
        - Records batch-level metrics
    """
    ctx = get_request_context(request)

    # Parse request body
    try:
        body = await request.json()
        api_req = BatchVLMRequest(**body)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in request body: {e!s}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e!s}",
        ) from e

    try:
        # Validate batch size
        if len(api_req.requests) > settings.batch.vlm_max_requests:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many requests in batch. Maximum allowed: {settings.batch.vlm_max_requests}",
            )

        # Validate all models are allowed for current hardware profile
        allowed = get_allowed_models()
        invalid_models = []
        for req in api_req.requests:
            if req.model and not is_model_allowed(req.model):
                invalid_models.append(req.model)
        if invalid_models:
            unique_invalid = sorted(set(invalid_models))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Models not supported on this hardware profile: {', '.join(unique_invalid)}. "
                    f"Allowed models: {', '.join(sorted(allowed))}"
                ),
            )

        # Convert API requests to domain entities
        domain_requests = [api_to_domain_vlm_request(req) for req in api_req.requests]

        # Execute batch
        logger.info(
            "batch_vlm_requested: request_id=%s, count=%d", ctx.request_id, len(domain_requests)
        )
        result = await use_case.execute(
            requests=domain_requests,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
            target_format=api_req.compression_format,
        )

        return BatchResponse(**result)

    except HTTPException:
        # Re-raise HTTPException
        raise
    except (InvalidRequestError, ValueError) as exc:
        logger.warning(
            "batch_vlm_validation_error: request_id=%s, error=%s", ctx.request_id, str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid batch VLM request: {exc!s}",
        ) from exc
    except Exception as exc:
        logger.exception(
            "unexpected_error_batch_vlm: request_id=%s, error_type=%s",
            ctx.request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred (request_id: {ctx.request_id}). Please try again later or contact support.",
        ) from exc

@router.post("/batch/chat/completions", tags=["Batch"], response_model=BatchResponse)
@limiter.limit("10/minute")
async def batch_chat_completions(
    request: Request,
    use_case: BatchChatUseCase = Depends(get_batch_chat_use_case),  # noqa: B008
) -> BatchResponse:
    """OpenAI-compatible batch text-only chat completion endpoint.

    Processes multiple OpenAI-compatible chat requests in parallel, returning all
    results in a single response. Rate limited to 10 requests per minute per IP address.

    Args:
        request: FastAPI Request object (injected). Body must contain
            BatchChatRequestOpenAI JSON with list of OpenAI-compatible chat requests (max 50).
        use_case: BatchChatUseCase instance (injected via DI).

    Returns:
        BatchResponse with OpenAI-compatible results for each request.

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Too many requests in batch (>50)
            - 500: Internal server error
    """
    ctx = get_request_context(request)

    # Parse request body (OpenAI-compatible format)
    try:
        body = await request.json()
        api_req = BatchChatRequestOpenAI(**body)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in request body: {e!s}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e!s}",
        ) from e

    try:
        # Validate batch size
        if len(api_req.requests) > settings.batch.chat_max_requests:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many requests in batch. Maximum allowed: {settings.batch.chat_max_requests}",
            )

        # Convert OpenAI-compatible requests to domain entities
        domain_requests = [
            api_to_domain_chat_request_openai(chat_req) for chat_req in api_req.requests
        ]

        # Execute batch processing
        logger.info(
            "batch_chat_completions_requested: request_id=%s, count=%d", ctx.request_id, len(domain_requests)
        )
        batch_result = await use_case.execute(
            requests=domain_requests,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
        )

        return BatchResponse(**batch_result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in batch_chat_completions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc!s}",
        ) from exc


@router.post("/batch/vlm/completions", tags=["Batch"], response_model=BatchResponse)
@limiter.limit("5/minute")
async def batch_vlm_completions(
    request: Request,
    use_case: BatchVLMUseCase = Depends(get_batch_vlm_use_case),  # noqa: B008
) -> BatchResponse:
    """OpenAI-compatible batch VLM chat completion endpoint.

    Processes multiple OpenAI-compatible VLM requests in parallel, with image compression
    and caching. Rate limited to 5 requests per minute per IP address due to resource intensity.

    Args:
        request: FastAPI Request object (injected). Body must contain
            BatchVLMRequestOpenAI JSON with list of OpenAI-compatible VLM requests (max 20).
        use_case: BatchVLMUseCase instance (injected via DI).

    Returns:
        BatchResponse with OpenAI-compatible results for each request.

    Raises:
        HTTPException: With appropriate status code for various error conditions:
            - 422: Invalid request body or validation error
            - 400: Too many requests in batch (>20) or invalid images
            - 500: Internal server error
    """
    ctx = get_request_context(request)

    # Parse request body (OpenAI-compatible format)
    try:
        body = await request.json()
        api_req = BatchVLMRequestOpenAI(**body)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in request body: {e!s}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e!s}",
        ) from e

    try:
        # Validate batch size
        if len(api_req.requests) > settings.batch.vlm_max_requests:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many requests in batch. Maximum allowed: {settings.batch.vlm_max_requests}",
            )

        # Convert OpenAI-compatible requests to domain entities
        domain_requests = [
            api_to_domain_vlm_request_openai(vlm_req) for vlm_req in api_req.requests
        ]

        # Execute batch processing
        logger.info(
            "batch_vlm_completions_requested: request_id=%s, count=%d", ctx.request_id, len(domain_requests)
        )
        batch_result = await use_case.execute(
            requests=domain_requests,
            client_ip=ctx.client_ip,
            project_name=ctx.project_name,
        )

        return BatchResponse(**batch_result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in batch_vlm_completions: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {exc!s}",
        ) from exc
