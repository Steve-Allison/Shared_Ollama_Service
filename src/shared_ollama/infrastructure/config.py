"""Centralized configuration management for Shared Ollama Service.

This module provides a single source of truth for all configuration values,
using pydantic-settings for environment variable loading and validation.

All configuration values can be set via environment variables with sensible defaults.
Supports .env files for local development.

Usage:
    from shared_ollama.infrastructure.config import settings

    # Access configuration values
    api_host = settings.api.host
    queue_max_concurrent = settings.queue.chat_max_concurrent
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaConfig(BaseSettings):
    """Ollama service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        case_sensitive=False,
        extra="ignore",
    )

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


class APIConfig(BaseSettings):
    """FastAPI server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="API server host")  # noqa: S104
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    reload: bool = Field(default=False, description="Enable auto-reload (development)")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", description="Logging level"
    )
    title: str = Field(default="Shared Ollama Service API", description="API title")
    version: str = Field(default="2.0.0", description="API version")
    docs_url: str = Field(default="/api/docs", description="OpenAPI docs URL")
    openapi_url: str = Field(default="/api/openapi.json", description="OpenAPI spec URL")


class QueueConfig(BaseSettings):
    """Request queue configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QUEUE_",
        case_sensitive=False,
        extra="ignore",
    )

    # Chat queue settings
    chat_max_concurrent: int = Field(
        default=6, ge=1, le=100, description="Max concurrent chat requests"
    )
    chat_max_queue_size: int = Field(
        default=50, ge=1, le=1000, description="Max chat queue depth"
    )
    chat_default_timeout: float = Field(
        default=60.0, ge=1.0, le=600.0, description="Default chat timeout (seconds)"
    )

    # VLM queue settings
    vlm_max_concurrent: int = Field(
        default=3, ge=1, le=50, description="Max concurrent VLM requests"
    )
    vlm_max_queue_size: int = Field(
        default=20, ge=1, le=500, description="Max VLM queue depth"
    )
    vlm_default_timeout: float = Field(
        default=120.0, ge=1.0, le=1200.0, description="Default VLM timeout (seconds)"
    )


class BatchConfig(BaseSettings):
    """Batch processing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BATCH_",
        case_sensitive=False,
        extra="ignore",
    )

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


class ImageProcessingConfig(BaseSettings):
    """Image processing configuration for VLM."""

    model_config = SettingsConfigDict(
        env_prefix="IMAGE_",
        case_sensitive=False,
        extra="ignore",
    )

    max_dimension: int = Field(
        default=1024, ge=256, le=2048, description="Max image dimension (pixels)"
    )
    jpeg_quality: int = Field(
        default=85, ge=1, le=100, description="JPEG compression quality"
    )
    png_compression: int = Field(
        default=6, ge=0, le=9, description="PNG compression level"
    )
    max_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024 * 1024,  # 1MB minimum
        le=100 * 1024 * 1024,  # 100MB maximum
        description="Max image size in bytes",
    )


class ImageCacheConfig(BaseSettings):
    """Image cache configuration."""

    model_config = SettingsConfigDict(
        env_prefix="IMAGE_CACHE_",
        case_sensitive=False,
        extra="ignore",
    )

    max_size: int = Field(
        default=100, ge=1, le=10000, description="Max cached images"
    )
    ttl_seconds: float = Field(
        default=3600.0, ge=60.0, le=86400.0, description="Cache TTL (seconds)"
    )


class ClientConfig(BaseSettings):
    """Async client configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CLIENT_",
        case_sensitive=False,
        extra="ignore",
    )

    timeout: int = Field(
        default=300, ge=1, le=3600, description="Request timeout (seconds)"
    )
    health_check_timeout: int = Field(
        default=5, ge=1, le=60, description="Health check timeout (seconds)"
    )
    max_connections: int = Field(
        default=50, ge=1, le=1000, description="Max HTTP connections"
    )
    max_keepalive_connections: int = Field(
        default=20, ge=1, le=500, description="Max keep-alive connections"
    )
    max_concurrent_requests: int | None = Field(
        default=None, ge=1, description="Max concurrent requests (None = unlimited)"
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Retry delay (seconds)"
    )
    verbose: bool = Field(default=False, description="Verbose logging")


class OllamaManagerConfig(BaseSettings):
    """Ollama manager configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_MANAGER_",
        case_sensitive=False,
        extra="ignore",
    )

    auto_detect_optimizations: bool = Field(
        default=True, description="Auto-detect system optimizations"
    )
    wait_for_ready: bool = Field(
        default=True, description="Wait for service to be ready on start"
    )
    max_wait_time: int = Field(
        default=30, ge=1, le=300, description="Max wait time for readiness (seconds)"
    )
    shutdown_timeout: int = Field(
        default=10, ge=1, le=60, description="Shutdown timeout (seconds)"
    )
    force_manage: bool = Field(
        default=True,
        description="Stop external Ollama instances and manage our own. If True, stops Homebrew/launchd services before starting.",
    )


class Settings(BaseSettings):
    """Root settings class containing all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    image: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    image_cache: ImageCacheConfig = Field(default_factory=ImageCacheConfig)
    client: ClientConfig = Field(default_factory=ClientConfig)
    ollama_manager: OllamaManagerConfig = Field(default_factory=OllamaManagerConfig)

    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls) -> Settings:
        """Get cached settings instance (singleton pattern)."""
        return cls()


# Global settings instance
settings = Settings.get_settings()

__all__ = [
    "APIConfig",
    "BatchConfig",
    "ClientConfig",
    "ImageCacheConfig",
    "ImageProcessingConfig",
    "OllamaConfig",
    "OllamaManagerConfig",
    "QueueConfig",
    "Settings",
    "settings",
]

