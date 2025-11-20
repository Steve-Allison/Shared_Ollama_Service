"""Batch routes for batch processing endpoints.

Provides batch endpoints for processing multiple chat and VLM requests
in parallel.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import Annotated

from shared_ollama.api.dependencies import (
    get_batch_chat_use_case,
    get_batch_vlm_use_case,
    get_request_context,
)
from shared_ollama.api.mappers import (
    api_to_domain_chat_request,
    api_to_domain_vlm_request,
)
from shared_ollama.api.middleware import limiter
from shared_ollama.api.models import (
    BatchChatRequest,
    BatchResponse,
    BatchVLMRequest,
)
from shared_ollama.application.batch_use_cases import (
    BatchChatUseCase,
    BatchVLMUseCase,
)
from shared_ollama.domain.exceptions import InvalidRequestError
from shared_ollama.infrastructure.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/batch/chat", tags=["Batch"], response_model=BatchResponse)
@limiter.limit("10/minute")
async def batch_chat(
    request: Request,
    use_case: Annotated[BatchChatUseCase, Depends(get_batch_chat_use_case)],
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
    use_case: Annotated[BatchVLMUseCase, Depends(get_batch_vlm_use_case)],
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
