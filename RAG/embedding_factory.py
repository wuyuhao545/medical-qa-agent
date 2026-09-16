# core/embeddings/embedding_factory.py
import threading
from functools import lru_cache
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import Field, ConfigDict, PrivateAttr

from config import EMBEDDING_MODEL
from core.cache.cache_manager import get_embedding_cache, EmbeddingCache
from utils.circuit_breaker import get_breaker

_OOM_MARKERS = ("out of memory", "cuda", "cublas", "cudnn")
_LOAD_FAIL_MARKERS = ("no such file", "can't load", "cannot load", "not found")


class CachedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cache: EmbeddingCache = Field(default_factory=get_embedding_cache)
    # HuggingFaceEmbeddings 是 Pydantic v2 的 BaseModel。在 Pydantic v2 中，以 _ 开头的属性必须用 PrivateAttr 声明
    _disabled: bool = PrivateAttr(default=False)
    _lock = PrivateAttr(default_factory=threading.Lock)          # 类变量，所有实例共享，OK

    def embed_query(self, text: str) -> List[float]:
        if self._disabled:
            raise RuntimeError("Embedding 已永久降级")

        cached = self.cache.get(text)
        if cached is not None:
            return cached

        cb = get_breaker("embedding", failure_threshold=5, recovery_timeout=30.0)
        try:
            vector = cb.call(super().embed_query, text)
        except Exception as e:
            msg = str(e).lower()
            if any(m in msg for m in _LOAD_FAIL_MARKERS):
                self._disabled = True
                print("❌ Embedding 加载失败，永久降级")
            elif any(m in msg for m in _OOM_MARKERS):
                cb.reset()   # OOM 不计入熔断
                print("⚠️ Embedding OOM，本次降级")
            raise
        self.cache.set(text, vector)
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List = [None] * len(texts)
        uncached_texts, uncached_idx = [], []
        for i, t in enumerate(texts):
            c = self.cache.get(t)
            if c is not None:
                results[i] = c
            else:
                uncached_texts.append(t)
                uncached_idx.append(i)

        if uncached_texts:
            cb = get_breaker("embedding", failure_threshold=5, recovery_timeout=30.0)
            try:
                vectors = cb.call(super().embed_documents, uncached_texts)
            except Exception as e:
                msg = str(e).lower()
                if any(m in msg for m in _LOAD_FAIL_MARKERS):
                    self._disabled = True
                    print("❌ Embedding 加载失败，永久降级")
                elif any(m in msg for m in _OOM_MARKERS):
                    cb.reset()
                    print("⚠️ Embedding 批处理 OOM，本次降级")
                raise
            for i, v in zip(uncached_idx, vectors):
                results[i] = v
                self.cache.set(texts[i], v)
        return results


@lru_cache(maxsize=1)
def get_embeddings():
    """单例：避免每次实例化都重新加载 m3e-large。"""
    return CachedHuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )