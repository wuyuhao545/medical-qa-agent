# core/cache/cache_manager.py
"""
缓存管理器：LLM 缓存 + 嵌入向量缓存
- EmbeddingCache：线程安全 LRU
- Redis 缓存：带熔断，失败自动降级内存
"""
import hashlib
import threading
from collections import OrderedDict
from typing import Optional, Dict, Any

from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, RedisCache

from utils.circuit_breaker import get_breaker
from config import EMBEDDING_CACHE_SIZE

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class EmbeddingCache:
    """线程安全 LRU 嵌入缓存。"""

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size
        self._hit_count = 0
        self._miss_count = 0

    def get(self, text: str) -> Optional[Any]:
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hit_count += 1
                return self._cache[key]
            self._miss_count += 1
            return None

    def set(self, text: str, vector: Any):
        key = self._hash(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = vector
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def stats(self) -> dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "size": len(self._cache),
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": self._hit_count / total if total else 0.0,
            }


def init_llm_cache(backend: str = "memory", redis_url: Optional[str] = None):
    if backend == "redis":
        if not REDIS_AVAILABLE:
            print("⚠️ 未安装 redis，降级为内存缓存")
            set_llm_cache(InMemoryCache())
            return
        cb = get_breaker("redis", failure_threshold=3, recovery_timeout=30.0)
        try:
            r = cb.call(redis.Redis.from_url, redis_url)
            set_llm_cache(RedisCache(redis_=r))
            print(f"✅ 已启用 Redis LLM 缓存: {redis_url}")
        except Exception as e:
            print(f"⚠️ Redis 不可用，降级为内存缓存: {e}")
            set_llm_cache(InMemoryCache())
    else:
        set_llm_cache(InMemoryCache())
        print("✅ 已启用内存 LLM 缓存")


_embedding_cache = EmbeddingCache(max_size=EMBEDDING_CACHE_SIZE)


def get_embedding_cache() -> EmbeddingCache:
    return _embedding_cache