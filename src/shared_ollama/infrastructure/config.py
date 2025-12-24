"""Configuration for Ollama service connection.

Simplified configuration module for pure Ollama client library.
Only contains OllamaConfig used by core utilities.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    """Ollama service configuration."""

    host: str = Field(default="localhost", description="Ollama service host")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama service port")
    base_url: str | None = Field(default=None, description="Full base URL (overrides host/port)")

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


__all__ = ["OllamaConfig"]
