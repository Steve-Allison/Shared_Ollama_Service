"""Centralized configuration management for Shared Ollama Service.

This module provides a single source of truth for all configuration values,
using TOML config files with pydantic validation. This is the infrastructure
layer's configuration module, complementing core/config.py.

Design Principles:
    - TOML Configuration: Loads from config.toml in project root
    - Pydantic Validation: Type-safe configuration with validation
    - Sensible Defaults: All settings have production-ready defaults
    - Singleton Pattern: Cached settings instance via lru_cache

Configuration Loading:
    1. Reads config.toml from project root (if exists)
    2. Merges with environment variables (with appropriate prefixes)
    3. Falls back to defaults if not set
    4. Validates all values using Pydantic

Configuration Sections:
    - OllamaConfig: Ollama service connection and settings
    - APIConfig: FastAPI server configuration
    - QueueConfig: Request queue settings (chat and VLM)
    - BatchConfig: Batch processing limits
    - ImageProcessingConfig: Image processing parameters
    - ImageCacheConfig: Image cache settings
    - ClientConfig: HTTP client configuration
    - OllamaManagerConfig: Ollama process management settings

Usage:
    from shared_ollama.infrastructure.config import settings

    # Access configuration values
    api_host = settings.api.host
    queue_max_concurrent = settings.queue.chat_max_concurrent
    ollama_url = settings.ollama.url

Note:
    This module is similar to core/config.py but uses TOML file loading
    instead of pure environment variables. Both modules provide the same
    configuration structure for consistency.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    """Ollama service configuration."""

    host: str = Field(default="localhost", description="Ollama service host")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama service port")
    base_url: str | None = Field(default=None, description="Full base URL (overrides host/port)")
    keep_alive: str = Field(default="5m", description="Model keep-alive duration")
    debug: bool = Field(default=False, description="Enable debug logging")
    metal: bool = Field(default=True, description="Enable Metal acceleration (Apple Silicon)")
    num_gpu: int = Field(default=-1, description="Number of GPU cores (-1 = all)")
    num_thread: int | None = Field(default=None, description="Number of CPU threads")
    max_ram: str | None = Field(default=None, description="Maximum RAM usage (e.g., '24GB')")
    num_parallel: int | None = Field(default=None, description="Number of parallel models")
    origins: str = Field(default="*", description="Allowed CORS origins")

    @property
    def url(self) -> str:
        """Get the full base URL for Ollama service."""
        if self.base_url:
            return self.base_url.rstrip("/")
        return f"http://{self.host}:{self.port}"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str | None) -> str | None:
        """Validate base URL format."""
        if v and not v.startswith(("http://", "https://")):
            msg = "base_url must start with http:// or https://"
            raise ValueError(msg)
        return v


class APIConfig(BaseModel):
    """FastAPI server configuration."""

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    reload: bool = Field(default=False, description="Enable auto-reload (development)")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", description="Logging level"
    )
    title: str = Field(default="Shared Ollama Service API", description="API title")
    version: str = Field(default="2.0.0", description="API version")
    docs_url: str = Field(default="/api/docs", description="OpenAPI docs URL")
    openapi_url: str = Field(default="/api/openapi.json", description="OpenAPI spec URL")


class QueueConfig(BaseModel):
    """Request queue configuration."""

    # Chat queue settings
    chat_max_concurrent: int = Field(
        default=6, ge=1, le=100, description="Max concurrent chat requests"
    )
    chat_max_queue_size: int = Field(default=50, ge=1, le=1000, description="Max chat queue depth")
    chat_default_timeout: float = Field(
        default=120.0, ge=1.0, le=600.0, description="Default chat timeout (seconds)"
    )

    # VLM queue settings
    vlm_max_concurrent: int = Field(
        default=3, ge=1, le=50, description="Max concurrent VLM requests"
    )
    vlm_max_queue_size: int = Field(default=20, ge=1, le=500, description="Max VLM queue depth")
    vlm_default_timeout: float = Field(
        default=150.0, ge=1.0, le=1200.0, description="Default VLM timeout (seconds)"
    )


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    chat_max_concurrent: int = Field(
        default=5, ge=1, le=50, description="Max concurrent batch chat requests"
    )
    chat_max_requests: int = Field(
        default=50, ge=1, le=1000, description="Max requests per batch chat"
    )
    vlm_max_concurrent: int = Field(
        default=3, ge=1, le=20, description="Max concurrent batch VLM requests"
    )
    vlm_max_requests: int = Field(
        default=20, ge=1, le=500, description="Max requests per batch VLM"
    )


class ImageProcessingConfig(BaseModel):
    """Image processing configuration for VLM."""

    max_dimension: int = Field(
        default=2667, ge=256, le=2667, description="Max image dimension (pixels)"
    )
    jpeg_quality: int = Field(default=85, ge=1, le=100, description="JPEG compression quality")
    png_compression: int = Field(default=6, ge=0, le=9, description="PNG compression level")
    max_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024 * 1024,  # 1MB minimum
        le=100 * 1024 * 1024,  # 100MB maximum
        description="Max image size in bytes",
    )


class ImageCacheConfig(BaseModel):
    """Image cache configuration."""

    max_size: int = Field(
        default=100, ge=1, le=10000, description="Max number of cached images (count, not bytes)"
    )
    ttl_seconds: float = Field(
        default=3600.0, ge=60.0, le=86400.0, description="Cache TTL (seconds)"
    )


class ResponseCacheConfig(BaseModel):
    """Response cache configuration."""

    enabled: bool = Field(default=True, description="Enable response caching")
    max_size: int = Field(
        default=1000, ge=1, le=100000, description="Max number of cached responses"
    )
    ttl_seconds: int = Field(
        default=3600, ge=60, le=86400, description="Cache TTL (seconds)"
    )
    similarity_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Minimum similarity for cache hits"
    )


class ClientConfig(BaseModel):
    """Async client configuration."""

    timeout: int = Field(default=180, ge=1, le=3600, description="Request timeout (seconds)")
    health_check_timeout: int = Field(
        default=5, ge=1, le=60, description="Health check timeout (seconds)"
    )
    max_connections: int = Field(default=50, ge=1, le=1000, description="Max HTTP connections")
    max_keepalive_connections: int = Field(
        default=20, ge=1, le=500, description="Max keep-alive connections"
    )
    max_concurrent_requests: int | None = Field(
        default=None, ge=1, description="Max concurrent requests (None = unlimited)"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Retry delay (seconds)")
    verbose: bool = Field(default=False, description="Verbose logging")


class OllamaManagerConfig(BaseModel):
    """Ollama manager configuration."""

    auto_detect_optimizations: bool = Field(
        default=True, description="Auto-detect system optimizations"
    )
    wait_for_ready: bool = Field(default=True, description="Wait for service to be ready on start")
    max_wait_time: int = Field(
        default=30, ge=1, le=300, description="Max wait time for readiness (seconds)"
    )
    shutdown_timeout: int = Field(default=10, ge=1, le=60, description="Shutdown timeout (seconds)")
    force_manage: bool = Field(
        default=True,
        description="Stop external Ollama instances and manage our own",
    )


class Settings(BaseModel):
    """Root settings class containing all configuration sections."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    image: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    image_cache: ImageCacheConfig = Field(default_factory=ImageCacheConfig)
    response_cache: ResponseCacheConfig = Field(default_factory=ResponseCacheConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    ollama_manager: OllamaManagerConfig = Field(default_factory=OllamaManagerConfig)

    @classmethod
    def from_toml(cls, config_path: Path | None = None) -> Settings:
        """Load settings from TOML file.

        Args:
            config_path: Path to config.toml. If None, searches for config.toml
                in project root.

        Returns:
            Settings instance populated from TOML file, or default settings
            if file not found.

        Raises:
            ValueError: If TOML file is invalid or contains validation errors.
        """
        if config_path is None:
            # Find project root (assumes this file is in src/shared_ollama/infrastructure/)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            config_path = project_root / "config.toml"

        if not config_path.exists():
            # Return default settings if no config file found
            return cls()

        try:
            with Path(config_path).open("rb") as f:
                config_data = tomllib.load(f)

            # Create nested config objects
            return cls(
                ollama=OllamaConfig(**config_data.get("ollama", {})),
                api=APIConfig(**config_data.get("api", {})),
                queue=QueueConfig(**config_data.get("queue", {})),
                batch=BatchConfig(**config_data.get("batch", {})),
                image=ImageProcessingConfig(**config_data.get("image", {})),
                image_cache=ImageCacheConfig(**config_data.get("image_cache", {})),
                response_cache=ResponseCacheConfig(**config_data.get("response_cache", {})),
                client=ClientConfig(**config_data.get("client", {})),
                ollama_manager=OllamaManagerConfig(**config_data.get("ollama_manager", {})),
            )
        except Exception as exc:
            msg = f"Failed to load config from {config_path}: {exc}"
            raise ValueError(msg) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings.from_toml()


# Global settings instance
settings = get_settings()

__all__ = [
    "APIConfig",
    "BatchConfig",
    "ClientConfig",
    "ImageCacheConfig",
    "ImageProcessingConfig",
    "OllamaConfig",
    "OllamaManagerConfig",
    "QueueConfig",
    "ResponseCacheConfig",
    "Settings",
    "get_settings",
    "settings",
]
