"""Reusable HTTPException builders for guardrail responses."""

from __future__ import annotations

from fastapi import HTTPException, status

from shared_ollama.api.limits import QUEUE_RETRY_AFTER_SECONDS


def queue_full_error() -> HTTPException:
    """Return an HTTPException for queue saturation."""

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "queue_full",
            "message": (
                "The shared generation queue is at capacity. "
                "Pause briefly, reduce concurrency, then retry."
            ),
            "retry_after_seconds": QUEUE_RETRY_AFTER_SECONDS,
        },
        headers={"Retry-After": str(QUEUE_RETRY_AFTER_SECONDS)},
    )


def queue_timeout_error() -> HTTPException:
    """Return an HTTPException for queue wait timeouts."""

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "request_timeout",
            "message": (
                "Timed out waiting for an available generation slot. "
                "Reduce prompt size or concurrent requests and try again."
            ),
            "retry_after_seconds": QUEUE_RETRY_AFTER_SECONDS,
        },
        headers={"Retry-After": str(QUEUE_RETRY_AFTER_SECONDS)},
    )


__all__ = ["queue_full_error", "queue_timeout_error"]

