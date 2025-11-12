"""
Structured logging utilities for the Shared Ollama Service.

All request/response events are written as JSON Lines to ``logs/requests.jsonl``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

REQUEST_LOGGER = logging.getLogger("ollama.requests")
if not REQUEST_LOGGER.handlers:
    REQUEST_LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(LOGS_DIR / "requests.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOGGER.addHandler(handler)
    REQUEST_LOGGER.propagate = False


def _json_default(value: Any) -> Any:
    """Fallback serializer for datetime and Path objects."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def log_request_event(event: dict[str, Any]) -> None:
    """
    Emit a structured request event.

    Args:
        event: Event payload. ``timestamp`` is injected automatically if absent.
    """
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(UTC).isoformat()
    REQUEST_LOGGER.info(json.dumps(event, default=_json_default))

