"""Core utility helpers for the Shared Ollama Service.

The utilities in this module are intentionally framework-agnostic and heavily
cached so they can be consumed by API routers, CLI scripts, and tests without
incurring expensive recomputation.  Highlights include:

* Project root detection that works for editable installs and wheel builds.
* Health-check helpers that delegate to the infrastructure layer.
* Centralized model-profile loading from ``config/models.yaml`` with strict
  validation and modern typing.
* Lazy imports for the synchronous client to avoid circular dependencies.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, cast

if TYPE_CHECKING:
    from shared_ollama.client.sync import SharedOllamaClient

try:
    import yaml
except ImportError:  # pragma: no cover - environments without YAML are unsupported
    yaml = None  # type: ignore[assignment]

from shared_ollama.infrastructure.config import OllamaConfig
from shared_ollama.infrastructure.health_checker import check_ollama_health

_ROOT_MARKERS: Final = ("pyproject.toml", ".git")
_MODELS_CONFIG_PATH: Final = Path("config/models.yaml")


class ModelConfigError(RuntimeError):
    """Raised when ``config/models.yaml`` is missing or malformed."""


class _ProfileConfig(TypedDict, total=False):
    name: str
    vlm_model: str
    text_model: str
    required_models: Sequence[str] | str
    warmup_models: Sequence[str] | str
    memory_hints: Mapping[str, int]
    min_ram_gb: int
    max_ram_gb: int | None
    largest_model_gb: int
    inference_buffer_gb: int
    service_overhead_gb: int


class _DefaultsConfig(TypedDict, total=False):
    inference_buffer_gb: int
    service_overhead_gb: int


class _ModelsConfig(TypedDict):
    profiles: list[_ProfileConfig]
    defaults: _DefaultsConfig


@dataclass(slots=True, frozen=True)
class ModelDefaults:
    """Strongly-typed view of the resolved model configuration."""

    vlm_model: str
    text_model: str
    required_models: tuple[str, ...]
    warmup_models: tuple[str, ...]
    memory_hints: Mapping[str, int]
    largest_model_gb: int
    inference_buffer_gb: int
    service_overhead_gb: int

    @property
    def allowed_models(self) -> frozenset[str]:
        bucket = {self.vlm_model, self.text_model}
        bucket.update(self.required_models)
        bucket.update(self.warmup_models)
        return frozenset(filter(None, bucket))


@functools.cache
def get_project_root() -> Path:
    """Return the repository root.

    Example
    -------
    >>> root = get_project_root()
    >>> (root / "pyproject.toml").exists()
    True
    """

    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return start.parents[3]


def get_ollama_base_url() -> str:
    """Return the Ollama base URL from the active configuration.

    Example
    -------
    >>> get_ollama_base_url().startswith("http")
    True
    """

    return OllamaConfig().url


def check_service_health(
    base_url: str | None = None,
    timeout: int = 5,
) -> tuple[Literal[True], None] | tuple[Literal[False], str]:
    """Perform a lightweight service health check.

    Example
    -------
    >>> healthy, _ = check_service_health(timeout=1)
    >>> isinstance(healthy, bool)
    True
    """

    target = base_url or get_ollama_base_url()
    return check_ollama_health(base_url=target, timeout=timeout)


def ensure_service_running(
    base_url: str | None = None,
    raise_on_fail: bool = True,
) -> bool:
    """Ensure the service is reachable, optionally raising on failure.

    Example
    -------
    >>> ensure_service_running(raise_on_fail=False)
    False
    """

    is_healthy, error = check_service_health(base_url)
    match (is_healthy, raise_on_fail):
        case (False, True):
            message = (
                f"Ollama service is not available. {error}\n"
                "Start the service with: ./scripts/core/start.sh"
            )
            raise ConnectionError(message)
        case _:
            return is_healthy


@functools.cache
def get_client_path() -> Path:
    """Return the absolute path to the synchronous client implementation."""

    return (get_project_root() / "src" / "shared_ollama" / "client" / "sync.py").resolve()


def import_client() -> type[SharedOllamaClient]:
    """Dynamically import the ``SharedOllamaClient`` class."""

    module = importlib.import_module("shared_ollama.client.sync")
    return module.SharedOllamaClient


@functools.cache
def _detect_system_info() -> tuple[str, int]:
    """Return ``(architecture, total_ram_gb)`` using lightweight heuristics."""

    arch = platform.machine()
    system = platform.system()
    total_ram_gb = 0

    match system:
        case "Darwin":
            with contextlib.suppress(Exception):
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    total_ram_gb = int(result.stdout.strip()) // (1024**3)
        case "Linux":
            with contextlib.suppress(OSError, ValueError), Path(
                "/proc/meminfo"
            ).open(encoding="utf-8") as meminfo:
                for line in meminfo:
                    if line.startswith("MemTotal:"):
                        total_ram_gb = int(line.split()[1]) // (1024**2)
                        break
        case _:
            pass

    return arch, total_ram_gb


def _coerce_int(value: Any, default: int) -> int:
    match value:
        case bool():
            return int(value)
        case int() | float():
            return int(value)
        case str():
            with contextlib.suppress(ValueError):
                return int(value.strip())
            return default
        case _:
            return default


def _normalize_models(value: object) -> tuple[str, ...]:
    match value:
        case None:
            return ()
        case str():
            return (value,)
        case Sequence() as seq:
            return tuple(str(item) for item in seq if item)
        case _:
            return ()


def _coerce_memory_hints(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(name): _coerce_int(hint, 0) for name, hint in value.items()}


def _read_models_config() -> _ModelsConfig:
    config_path = get_project_root() / _MODELS_CONFIG_PATH
    if not config_path.exists():
        raise ModelConfigError(f"Model configuration file not found: {config_path}")
    if yaml is None:
        raise ModelConfigError("PyYAML is required to read config/models.yaml")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ModelConfigError("config/models.yaml must be a mapping")

    profiles_obj = raw.get("profiles")
    if not isinstance(profiles_obj, list) or not profiles_obj:
        raise ModelConfigError("config/models.yaml must define a non-empty 'profiles' list")

    profiles: list[_ProfileConfig] = []
    for profile in profiles_obj:
        if not isinstance(profile, dict):
            raise ModelConfigError("Each profile entry must be a mapping")
        profiles.append(cast(_ProfileConfig, profile))

    defaults_obj = raw.get("defaults") or {}
    if not isinstance(defaults_obj, dict):
        raise ModelConfigError("defaults section must be a mapping")

    return {"profiles": profiles, "defaults": cast(_DefaultsConfig, defaults_obj)}


def _select_profile(profiles: Sequence[_ProfileConfig], ram_gb: int) -> _ProfileConfig:
    for profile in profiles:
        min_ram = _coerce_int(profile.get("min_ram_gb"), 0)
        max_ram = profile.get("max_ram_gb")
        if max_ram is None and ram_gb >= min_ram:
            return profile
        if max_ram is not None and min_ram <= ram_gb <= _coerce_int(max_ram, ram_gb):
            return profile
    return profiles[-1]


def _build_model_defaults(
    selected: _ProfileConfig,
    defaults_section: _DefaultsConfig,
) -> ModelDefaults:
    vlm_model = str(selected.get("vlm_model", "")).strip()
    text_model = str(selected.get("text_model", "")).strip()
    if not vlm_model or not text_model:
        raise ModelConfigError("Each profile must declare both vlm_model and text_model")

    pick = selected.get
    fallback = defaults_section.get

    return ModelDefaults(
        vlm_model=vlm_model,
        text_model=text_model,
        required_models=_normalize_models(selected.get("required_models")),
        warmup_models=_normalize_models(selected.get("warmup_models")),
        memory_hints=_coerce_memory_hints(selected.get("memory_hints")),
        largest_model_gb=_coerce_int(pick("largest_model_gb"), 8),
        inference_buffer_gb=_coerce_int(
            pick("inference_buffer_gb", fallback("inference_buffer_gb", 4)),
            4,
        ),
        service_overhead_gb=_coerce_int(
            pick("service_overhead_gb", fallback("service_overhead_gb", 2)),
            2,
        ),
    )


@functools.cache
def _load_model_profile_defaults() -> ModelDefaults:
    config = _read_models_config()
    _, ram = _detect_system_info()
    selected_profile = _select_profile(config["profiles"], ram or 32)
    return _build_model_defaults(selected_profile, config["defaults"])


@functools.cache
def get_default_vlm_model() -> str:
    """Return the default VLM model name for the active hardware profile."""

    return _load_model_profile_defaults().vlm_model


@functools.cache
def get_default_text_model() -> str:
    """Return the default text model name for the active hardware profile."""

    return _load_model_profile_defaults().text_model


@functools.cache
def get_warmup_models() -> list[str]:
    """Return the list of models that should be pre-warmed at startup."""

    return list(_load_model_profile_defaults().warmup_models)


@functools.cache
def get_allowed_models() -> set[str]:
    """Return the set of models allowed on the current machine."""

    return set(_load_model_profile_defaults().allowed_models)


def is_model_allowed(model_name: str | None) -> bool:
    """Return ``True`` when the requested model is permitted for this host."""

    match model_name:
        case None:
            return True
        case _:
            return model_name in get_allowed_models()


__all__ = [
    "ModelConfigError",
    "ModelDefaults",
    "check_service_health",
    "ensure_service_running",
    "get_allowed_models",
    "get_client_path",
    "get_default_text_model",
    "get_default_vlm_model",
    "get_ollama_base_url",
    "get_project_root",
    "get_warmup_models",
    "import_client",
    "is_model_allowed",
]
