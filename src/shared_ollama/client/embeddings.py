"""Embeddings client methods for Ollama API.

Provides embeddings functionality for generating vector embeddings from text.
Embeddings are useful for semantic search, RAG systems, and similarity matching.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)

def embeddings_sync(
    session: requests.Session,
    base_url: str,
    prompt: str,
    model: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Generate embeddings for a text prompt (synchronous).

    Args:
        session: requests.Session instance for HTTP requests.
        base_url: Base URL for Ollama service.
        prompt: Text prompt to generate embeddings for.
        model: Model name to use for embeddings.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary containing:
            - embedding: List of float values (embedding vector)
            - model: Model name used
            - prompt: Original prompt text

    Raises:
        ValueError: If response is not a dictionary or missing embedding field.
        json.JSONDecodeError: If response is not valid JSON.
        requests.exceptions.HTTPError: If HTTP request fails.
        requests.exceptions.RequestException: If network error occurs.
    """
    payload = {
        "model": model,
        "prompt": prompt,
    }

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        response = session.post(
            f"{base_url}/api/embeddings",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict):
            msg = f"Expected dict response, got {type(data).__name__}"
            raise ValueError(msg)

        if "embedding" not in data:
            msg = "Response missing 'embedding' field"
            raise ValueError(msg)


        return data

    except json.JSONDecodeError as exc:
        logger.exception("Failed to decode JSON response from /api/embeddings for %s", model)
        raise
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else None
        logger.exception("HTTP error generating embeddings with %s: %s", model, status_code)
        raise
    except requests.exceptions.RequestException as exc:
        logger.exception("Request error generating embeddings with %s: %s", model, exc)
        raise

