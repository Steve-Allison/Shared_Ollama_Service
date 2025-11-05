"""Type stubs for utils module."""

from pathlib import Path
from typing import Type

def get_ollama_base_url() -> str: ...

def check_service_health(
    base_url: str | None = ...,
    timeout: int = ...,
) -> tuple[bool, str | None]: ...

def ensure_service_running(
    base_url: str | None = ...,
    raise_on_fail: bool = ...,
) -> bool: ...

def get_project_root() -> Path | None: ...

def get_client_path() -> Path | None: ...

def import_client() -> Type: ...

