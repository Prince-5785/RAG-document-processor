"""
Cache Service for InsureQuery.
Provides Redis-backed embedding caching with safe float32 binary serialization,
TTL support, and graceful fallback to an in-memory LRU cache when Redis is unavailable.
"""

import os
import hashlib
import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheService:
    """Embedding cache supporting Redis with in-memory fallback."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('cache', {})
        self.enabled = self.config.get('enable_embedding_cache', True)
        self.redis_url = os.environ.get('REDIS_URL') or self.config.get('redis_url', 'redis://localhost:6379/0')
        self.ttl_seconds = int(self.config.get('ttl_seconds', 86400))
        self.logger = logging.getLogger(__name__)

        self.redis_client = None
        self.is_redis_connected = False
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._max_memory_items = 10000

        self.stats = {
            'hits': 0,
            'misses': 0,
            'backend': 'none'
        }

        if self.enabled:
            self._init_backend()

    def _init_backend(self) -> None:
        """Attempts to initialize Redis client; falls back to in-memory cache."""
        requested_backend = self.config.get('backend', 'redis')
        if requested_backend != 'in_memory' and REDIS_AVAILABLE and self.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                    decode_responses=False
                )
                self.redis_client.ping()
                self.is_redis_connected = True
                self.stats['backend'] = 'redis'
                self.logger.info(f"Connected to Redis cache at {self.redis_url}")
                return
            except Exception as e:
                self.logger.warning(f"Could not connect to Redis ({e}). Falling back to in-memory cache.")
                self.redis_client = None
                self.is_redis_connected = False

        self.stats['backend'] = 'in_memory'
        self.logger.info("Initialized in-memory embedding cache fallback.")

    def _generate_key(self, model_name: str, text: str) -> str:
        """Generate a deterministic namespaced key based on model and text hash."""
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        sanitized_model = model_name.replace('/', '_')
        return f"insurequery:emb:{sanitized_model}:{text_hash}"

    def get_embedding(self, model_name: str, text: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if present.
        Returns a 1D numpy array or None.
        """
        if not self.enabled or not text:
            return None

        key = self._generate_key(model_name, text)

        # Try Redis first
        if self.is_redis_connected and self.redis_client:
            try:
                raw_bytes = self.redis_client.get(key)
                if raw_bytes is not None:
                    embedding = np.frombuffer(raw_bytes, dtype=np.float32)
                    self.stats['hits'] += 1
                    return embedding
            except Exception as e:
                self.logger.warning(f"Redis get error: {e}. Switching to memory cache.")
                self.is_redis_connected = False

        # Fallback to memory
        if key in self._memory_cache:
            self.stats['hits'] += 1
            return self._memory_cache[key]

        self.stats['misses'] += 1
        return None

    def set_embedding(self, model_name: str, text: str, embedding: np.ndarray, ttl: Optional[int] = None) -> None:
        """
        Store embedding in cache with safe binary float32 serialization.
        """
        if not self.enabled or embedding is None or len(embedding) == 0:
            return

        key = self._generate_key(model_name, text)
        arr = np.asarray(embedding, dtype=np.float32)
        ttl_to_use = ttl or self.ttl_seconds

        # Store in Redis if connected
        if self.is_redis_connected and self.redis_client:
            try:
                raw_bytes = arr.tobytes()
                self.redis_client.set(key, raw_bytes, ex=ttl_to_use)
                return
            except Exception as e:
                self.logger.warning(f"Redis set error: {e}. Storing in memory instead.")
                self.is_redis_connected = False

        # Memory cache with simple eviction
        if len(self._memory_cache) >= self._max_memory_items:
            # Drop oldest 1000 items
            keys_to_remove = list(self._memory_cache.keys())[:1000]
            for k in keys_to_remove:
                self._memory_cache.pop(k, None)

        self._memory_cache[key] = arr

    def clear(self) -> None:
        """Clear cache entries."""
        self._memory_cache.clear()
        if self.is_redis_connected and self.redis_client:
            try:
                keys = self.redis_client.keys("insurequery:emb:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Failed to clear Redis keys: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and usage statistics."""
        return {
            'enabled': self.enabled,
            'backend': 'redis' if self.is_redis_connected else ('in_memory' if self.enabled else 'disabled'),
            'redis_connected': self.is_redis_connected,
            'memory_items': len(self._memory_cache),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': (
                round(self.stats['hits'] / (self.stats['hits'] + self.stats['misses']), 3)
                if (self.stats['hits'] + self.stats['misses']) > 0 else 0.0
            )
        }
