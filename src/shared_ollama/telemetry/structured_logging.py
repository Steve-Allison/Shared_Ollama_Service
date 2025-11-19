"""Structured logging utilities for the Shared Ollama Service.

This module provides JSON-based structured logging for request/response
events. All events are written as JSON Lines (JSONL) format to a log file.

Key behaviors:
    - JSON Lines format (one JSON object per line)
    - Automatic timestamp injection if not present
    - Custom serialization for datetime and Path objects
    - Cached log directory resolution for performance
    - Non-propagating logger to avoid duplicate logs

Log file:
    - Location: ``logs/requests.jsonl`` (relative to project root)
    - Format: One JSON object per line
    - Encoding: UTF-8
    - Rotation: Not implemented (file grows unbounded)
"""

from __future__ import annotations

import functools
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter


# Cache log directory resolution
@functools.cache
def _get_logs_dir() -> Path:
    """Get logs directory with caching for performance.

    Resolves the logs directory relative to the project root and creates
    it if it doesn't exist. Result is cached since the path doesn't change
    at runtime.

    Returns:
        Path to logs directory.

    Side effects:
        Creates logs directory if it doesn't exist.
    """
    logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


LOGS_DIR = _get_logs_dir()

REQUEST_LOGGER = logging.getLogger("ollama.requests")
if not REQUEST_LOGGER.handlers:
    REQUEST_LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(LOGS_DIR / "requests.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOGGER.addHandler(handler)
    REQUEST_LOGGER.propagate = False


def _json_default(value: Any) -> Any:
    """Fallback serializer for datetime and Path objects.

    Uses Pydantic's built-in serialization for datetime objects, which handles
    timezone-aware datetimes correctly. Falls back to string conversion for
    Path and other types.

    Args:
        value: Value to serialize. Expected to be datetime, Path, or
            other non-serializable type.

    Returns:
        Serialized value:
            - datetime: ISO 8601 format string (via Pydantic)
            - Path: String representation of path
            - Other: String representation

    Side effects:
        None. Pure function.
    """
    match value:
        case datetime():
            # Use Pydantic's datetime serialization for proper timezone handling
            return TypeAdapter(datetime).dump_python(value, mode="json")
        case Path():
            return str(value)
        case _:
            return str(value)


def log_request_event(event: dict[str, Any]) -> None:
    """Emit a structured request event.

    Writes a JSON-formatted log entry to the requests log file. Automatically
    injects timestamp if not present in the event dictionary.

    Args:
        event: Event payload dictionary. Should contain:
            - event: str - Event type (e.g., "api_request", "ollama_request")
            - operation: str - Operation name (e.g., "generate", "chat")
            - status: str - Status ("success", "error")
            - Additional fields as needed (model, latency_ms, etc.)
        The 'timestamp' field is automatically added if missing.

    Side effects:
        - Writes JSON line to logs/requests.jsonl file
        - Adds timestamp to event dictionary if missing
        - May raise IOError if file write fails

    Example:
        >>> log_request_event({
        ...     "event": "api_request",
        ...     "operation": "generate",
        ...     "status": "success",
        ...     "model": "qwen2.5vl:7b",
        ...     "latency_ms": 1234.56
        ... })
    """
    event.setdefault("timestamp", datetime.now(UTC).isoformat())
    REQUEST_LOGGER.info(json.dumps(event, default=_json_default))


__all__ = ["log_request_event"]
