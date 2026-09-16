# core/retrievers/rerank.py
from typing import Any, Optional, Dict

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever

from config import RERANKER_MODEL   # 建议挪到 config.py
from utils.circuit_breaker import get_breaker, CircuitOpenError

_MODEL_CACHE: Dict[str, HuggingFaceCrossEncoder] = {}


def _get_cross_encoder(model_name: str) -> HuggingFaceCrossEncoder:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = HuggingFaceCrossEncoder(model_name=model_name)
    return _MODEL_CACHE[model_name]


def build_reranker(model_name=RERANKER_MODEL, top_n=None):
    if top_n is None:
        top_n = 1000
    elif top_n <= 0:
        raise ValueError(f"top_n 必须为正整数，收到 {top_n}")
    cross_encoder = _get_cross_encoder(model_name)
    return CrossEncoderReranker(model=cross_encoder, top_n=top_n)


def wrap_with_reranker(
    base_retriever: Any,
    model_name: str = RERANKER_MODEL,
    top_n: Optional[int] = None,
) -> Any:
    cb = get_breaker("reranker", failure_threshold=3, recovery_timeout=60.0)
    try:
        compressor = cb.call(build_reranker, model_name, top_n=top_n)
    except (CircuitOpenError, Exception) as e:
        print(f"⚠️ Reranker 不可用，跳过热排: {e}")
        return base_retriever
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )