"""Structured logging utilities for the Shared Ollama Service.

This module provides JSON-based structured logging for request/response
events. All events are written as JSON Lines (JSONL) format to a log file
for easy parsing and analysis.

Key Features:
    - JSON Lines Format: One JSON object per line for easy parsing
    - Automatic Timestamps: Injected if not present in event data
    - Custom Serialization: Handles datetime and Path objects correctly
    - Performance: Cached log directory resolution
    - Isolation: Non-propagating logger to avoid duplicate logs

Log File Configuration:
    - Location: ``logs/requests.jsonl`` (relative to project root)
    - Format: JSON Lines (one JSON object per line)
    - Encoding: UTF-8
    - Rotation: Not implemented (file grows unbounded - consider log rotation
      for production)

Design Principles:
    - Structured Data: All events are JSON objects with consistent schema
    - Performance: Minimal overhead, cached paths, efficient serialization
    - Reliability: Errors in logging don't affect request processing
    - Observability: Comprehensive event data for analysis

Event Schema:
    All events should include:
        - event: Event type identifier (e.g., "api_request", "http_request")
        - timestamp: ISO 8601 timestamp (auto-injected if missing)
        - Additional fields: Operation-specific metadata (request_id, model,
          latency_ms, status, etc.)
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

    Writes a JSON-formatted log entry to the requests log file in JSON Lines
    format. Automatically injects timestamp if not present in the event dictionary.
    Handles custom serialization for datetime and Path objects.

    Args:
        event: Event payload dictionary. Should contain:
            - event: str - Event type identifier (e.g., "api_request", "http_request",
              "ollama_request")
            - operation: str - Operation name (e.g., "generate", "chat", "vlm")
            - status: str - Status ("success" or "error")
            - Additional fields as needed:
                - request_id: Unique request identifier
                - model: Model name used
                - latency_ms: Request latency in milliseconds
                - client_ip: Client IP address
                - project_name: Project identifier
                - error_type: Error type (if status="error")
                - error_message: Error message (if status="error")
        The 'timestamp' field is automatically added if missing (ISO 8601 format).

    Side Effects:
        - Writes JSON line to logs/requests.jsonl file
        - Adds timestamp to event dictionary if missing (mutates input dict)
        - May raise IOError if file write fails (rare, but possible)

    Note:
        This function is designed to be non-blocking and fast. Errors during
        logging are caught and logged to Python's logging system but don't
        affect request processing. The function mutates the input event dict
        by adding a timestamp if missing.

    Example:
        >>> log_request_event({
        ...     "event": "api_request",
        ...     "operation": "generate",
        ...     "status": "success",
        ...     "model": "qwen3:14b-q4_K_M",
        ...     "latency_ms": 1234.56
        ... })
    """
    event.setdefault("timestamp", datetime.now(UTC).isoformat())
    REQUEST_LOGGER.info(json.dumps(event, default=_json_default))


__all__ = ["log_request_event"]
