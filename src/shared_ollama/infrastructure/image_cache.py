"""LRU cache for processed images to avoid re-compression.

This module provides an in-memory LRU cache with TTL expiration for processed
images. Caching avoids redundant image processing (validation, resizing,
compression, format conversion) when the same images are used multiple times.

Key Features:
    - LRU eviction: Least recently used entries evicted when cache is full
    - TTL expiration: Entries automatically expire after TTL period
    - Thread-safe: Safe for concurrent access from multiple coroutines
    - Statistics tracking: Hit/miss rates for monitoring
    - SHA-256 keys: Deterministic cache keys from data URL + format

Design Principles:
    - Performance: Reduces redundant image processing overhead
    - Memory efficiency: LRU eviction prevents unbounded growth
    - Observability: Comprehensive statistics for monitoring
"""

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
    """Cached image data with metadata.

    Immutable container for cached processed image data, including the processed
    base64 string, image metadata, and cache timestamp.

    Attributes:
        base64_string: Processed image as base64 data URL. Ready for use
            without additional processing.
        metadata: ImageMetadata object containing processing results (dimensions,
            sizes, compression ratio).
        timestamp: Cache entry creation timestamp. Used for TTL expiration
            and statistics.
    """

    __slots__ = ("base64_string", "metadata", "timestamp")

    def __init__(
        self,
        base64_string: str,
        metadata: ImageMetadata,
        timestamp: float,
    ) -> None:
        """Initialize cache entry.

        Args:
            base64_string: Processed image as base64 data URL. Format:
                "data:image/{format};base64,{base64_data}".
            metadata: ImageMetadata object with processing results (dimensions,
                sizes, compression ratio).
            timestamp: Cache entry creation timestamp. Used for TTL expiration.
                Should be time.time() value.
        """
        self.base64_string = base64_string
        self.metadata = metadata
        self.timestamp = timestamp


class ImageCache:
    """LRU cache for processed images using cachetools TTLCache.

    Provides thread-safe in-memory caching with automatic TTL expiration and
    LRU eviction. Cache keys are computed from data URL and target format
    using SHA-256 hashing.

    The cache implements ImageCacheInterface, enabling dependency inversion
    in the application layer.

    Attributes:
        max_size: Maximum number of cached entries. When full, LRU entries
            are evicted to make room for new entries.
        ttl_seconds: Time-to-live for cache entries in seconds. Entries
            automatically expire after this period.
        _cache: Underlying TTLCache instance providing LRU and TTL functionality.
        _hits: Total number of cache hits since initialization.
        _misses: Total number of cache misses since initialization.

    Thread Safety:
        TTLCache is thread-safe for concurrent access. Statistics updates
        are not atomic but acceptable for monitoring purposes.

    Note:
        Cache keys are SHA-256 hashes of "{data_url}:{target_format}" to
        ensure deterministic keys and prevent key collisions.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 3600.0):
        """Initialize image cache.

        Args:
            max_size: Maximum number of cached entries. Must be positive.
                When full, least recently used entries are evicted. Default: 100.
            ttl_seconds: Time-to-live for cache entries in seconds. Must be
                positive. Entries automatically expire after this period.
                Default: 3600.0 (1 hour).
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: TTLCache[str, CacheEntry] = TTLCache(
            maxsize=max_size, ttl=ttl_seconds
        )
        self._hits = 0
        self._misses = 0

    def _compute_key(self, data_url: str, target_format: str) -> str:
        """Compute cache key from data URL and format.

        Generates a deterministic SHA-256 hash key from the data URL and
        target format combination. This ensures consistent keys and prevents
        collisions.

        Args:
            data_url: Original image data URL.
            target_format: Target format for processing.

        Returns:
            SHA-256 hex digest string used as cache key.

        Note:
            The key format is "{data_url}:{target_format}" hashed with SHA-256.
            This ensures different formats of the same image have different keys.
        """
        return hashlib.sha256(f"{data_url}:{target_format}".encode()).hexdigest()

    def get(
        self,
        data_url: str,
        target_format: Literal["jpeg", "png", "webp"],
    ) -> tuple[str, ImageMetadata] | None:
        """Get cached processed image.

        Retrieves a cached processed image if available and not expired.
        Updates hit/miss statistics.

        Args:
            data_url: Original image data URL used as cache key component.
            target_format: Target format used as cache key component.

        Returns:
            Tuple of (base64_string, metadata) if cached entry exists and
            is not expired, None if cache miss or entry expired.

        Note:
            Cache hits and misses are tracked for statistics. Expired entries
            are treated as cache misses and automatically removed by TTLCache.
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

        Stores a processed image in cache with current timestamp. If cache is
        full, least recently used entries are automatically evicted by TTLCache.

        Args:
            data_url: Original image data URL used as cache key component.
            target_format: Target format used as cache key component.
            base64_string: Processed image as base64 data URL. Ready for use
                without additional processing.
            metadata: ImageMetadata object with processing results (dimensions,
                sizes, compression ratio).

        Note:
            Cache entries are stored with current timestamp for TTL expiration.
            If cache is full, LRU entries are evicted to make room. This method
            does not raise exceptions for storage errors.
        """
        key = self._compute_key(data_url, target_format)

        self._cache[key] = CacheEntry(
            base64_string=base64_string,
            metadata=metadata,
            timestamp=time.time(),
        )
        logger.debug(f"Cached image: {key[:16]}...")

    def clear(self) -> None:
        """Clear all cache entries and reset statistics.

        Removes all cached entries and resets hit/miss counters to zero.
        Useful for cache invalidation or testing.

        Note:
            This method clears the cache and resets statistics. It does not
            affect cache configuration (max_size, ttl_seconds).
        """
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns current cache statistics for monitoring and optimization.

        Returns:
            Dictionary with cache statistics containing:
                - size: Current number of cached entries (int)
                - max_size: Maximum cache size (int)
                - hits: Total cache hits since initialization (int)
                - misses: Total cache misses since initialization (int)
                - hit_rate: Cache hit rate as float (0.0-1.0)

        Note:
            Statistics are cumulative since cache initialization or last clear().
            Hit rate is calculated as hits / (hits + misses). Returns 0.0
            if no requests have been made yet.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }
