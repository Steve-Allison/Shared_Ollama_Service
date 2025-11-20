"""LRU cache for processed images to avoid re-compression."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Literal

from cachetools import TTLCache

if TYPE_CHECKING:
    from shared_ollama.infrastructure.image_processing import ImageMetadata

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cached image data with metadata."""

    __slots__ = ("base64_string", "metadata", "timestamp")

    def __init__(
        self,
        base64_string: str,
        metadata: ImageMetadata,
        timestamp: float,
    ) -> None:
        """Initialize cache entry.

        Args:
            base64_string: Processed base64 string.
            metadata: Image metadata.
            timestamp: Cache timestamp.
        """
        self.base64_string = base64_string
        self.metadata = metadata
        self.timestamp = timestamp


class ImageCache:
    """LRU cache for processed images using cachetools TTLCache.

    Provides thread-safe caching with automatic TTL expiration.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600.0):
        """Initialize image cache.

        Args:
            max_size: Maximum number of cached images.
            ttl_seconds: Time-to-live for cache entries.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: TTLCache[str, CacheEntry] = TTLCache(
            maxsize=max_size, ttl=ttl_seconds
        )
        self._hits = 0
        self._misses = 0

    def _compute_key(self, data_url: str, target_format: str) -> str:
        """Compute cache key from data URL and format."""
        return hashlib.sha256(f"{data_url}:{target_format}".encode()).hexdigest()

    def get(
        self,
        data_url: str,
        target_format: Literal["jpeg", "png", "webp"],
    ) -> tuple[str, ImageMetadata] | None:
        """Get cached processed image.

        Args:
            data_url: Original data URL.
            target_format: Target format.

        Returns:
            Tuple of (base64_string, metadata) if cached and valid, None otherwise.
        """
        key = self._compute_key(data_url, target_format)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        self._hits += 1
        logger.debug(f"Cache hit: {key[:16]}...")
        return entry.base64_string, entry.metadata

    def put(
        self,
        data_url: str,
        target_format: Literal["jpeg", "png", "webp"],
        base64_string: str,
        metadata: ImageMetadata,
    ) -> None:
        """Cache processed image.

        Args:
            data_url: Original data URL.
            target_format: Target format.
            base64_string: Processed base64 string.
            metadata: Image metadata.
        """
        key = self._compute_key(data_url, target_format)

        self._cache[key] = CacheEntry(
            base64_string=base64_string,
            metadata=metadata,
            timestamp=time.time(),
        )
        logger.debug(f"Cached image: {key[:16]}...")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }
