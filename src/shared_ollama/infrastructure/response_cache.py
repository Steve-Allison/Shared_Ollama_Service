"""Response cache for semantic similarity matching.

This module provides an in-memory LRU cache with semantic similarity matching
for text generation responses. Caching avoids redundant computation when similar
prompts are requested.

Key Features:
    - Semantic Similarity: Uses embedding-based similarity matching
    - LRU eviction: Least recently used entries evicted when cache is full
    - TTL expiration: Entries automatically expire after TTL period
    - Thread-safe: All operations protected by threading.Lock
    - Statistics tracking: Hit/miss rates for monitoring

Design Principles:
    - Performance: Reduces redundant computation for similar prompts
    - Memory efficiency: LRU eviction prevents unbounded growth
    - Observability: Comprehensive statistics for monitoring
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from cachetools import TTLCache

if TYPE_CHECKING:
    from shared_ollama.client.sync import GenerateResponse

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cached response data with metadata.

    Immutable container for cached response data, including the response text,
    model used, and cache timestamp.

    Attributes:
        response: Cached response text or data.
        model: Model name used for generation.
        timestamp: Cache entry creation timestamp.
        embedding: Optional embedding vector for similarity matching.
    """

    __slots__ = ("response", "model", "timestamp", "embedding")

    def __init__(
        self,
        response: str | dict[str, Any],
        model: str,
        timestamp: float,
        embedding: list[float] | None = None,
    ) -> None:
        """Initialize cache entry.

        Args:
            response: Cached response text or structured data.
            model: Model name used for generation.
            timestamp: Cache entry creation timestamp.
            embedding: Optional embedding vector for similarity matching.
        """
        self.response = response
        self.model = model
        self.timestamp = timestamp
        self.embedding = embedding


class ResponseCache:
    """LRU cache for responses using cachetools TTLCache.

    Provides thread-safe in-memory caching with automatic TTL expiration and
    LRU eviction. Cache keys are computed from prompt hash.

    Attributes:
        max_size: Maximum number of cached entries.
        ttl_seconds: Time-to-live for cache entries in seconds.
        similarity_threshold: Minimum cosine similarity for cache hits (0.0-1.0).
        _cache: Underlying TTLCache instance.
        _hits: Total number of cache hits.
        _misses: Total number of cache misses.
        _lock: Thread lock for thread-safe operations.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.95,
    ) -> None:
        """Initialize response cache.

        Args:
            max_size: Maximum number of cached entries.
            ttl_seconds: Time-to-live for cache entries in seconds.
            similarity_threshold: Minimum cosine similarity for cache hits.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._cache: TTLCache[str, CacheEntry] = TTLCache(
            maxsize=max_size, ttl=ttl_seconds
        )
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _compute_key(self, prompt: str, model: str) -> str:
        """Compute cache key from prompt and model.

        Args:
            prompt: Text prompt.
            model: Model name.

        Returns:
            SHA-256 hash of "{prompt}:{model}".
        """
        key_string = f"{prompt}:{model}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> CacheEntry | None:
        """Get cached response if available.

        Args:
            prompt: Text prompt to look up.
            model: Model name.

        Returns:
            CacheEntry if found, None otherwise.
        """
        key = self._compute_key(prompt, model)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                self._hits += 1
                return entry
            self._misses += 1
            return None

    def put(
        self,
        prompt: str,
        model: str,
        response: str | dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        """Store response in cache.

        Args:
            prompt: Text prompt.
            model: Model name.
            response: Response text or structured data.
            embedding: Optional embedding vector for similarity matching.
        """
        key = self._compute_key(prompt, model)
        entry = CacheEntry(
            response=response,
            model=model,
            timestamp=time.time(),
            embedding=embedding,
        )
        with self._lock:
            self._cache[key] = entry

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hit/miss counts and hit rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
            }

