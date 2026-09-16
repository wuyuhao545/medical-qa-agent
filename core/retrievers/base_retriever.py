# core/retrievers/base_retriever.py
"""
基础检索器：BM25（稀疏）+ 向量（稠密）。
职责边界：
  - 本模块只提供"构建原子检索器"和"组合检索器"的能力。
  - 不涉及 HyDE（由工具层单独用 hyde.create_hyde_retriever 构建后再组合）。
  - 不读 config，不选深度，所有 k 由调用方传入。
对外暴露：
  - build_bm25_retriever(chunks, k)                 单路：BM25
  - build_vector_retriever(vector_store, k)         单路：纯向量
  - build_ensemble_retriever(bm25, vector, ...)     通用：组合任意两路
  - build_base_retriever(vector_store, chunks, ...) 便捷：BM25 + 纯向量
"""

from typing import Any, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 默认配置 
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_RRF_C = 60

# 单路构建：BM25 
def build_bm25_retriever(
    chunks: List[Document],
    k: int = 10,
) -> BM25Retriever:
    """从切好的文档块构建 BM25 稀疏检索器。"""
    if not chunks:
        raise ValueError("build_bm25_retriever 需要非空的 chunks")
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever

# 单路构建：纯向量
def build_vector_retriever(
    vector_store: Any,
    k: int = 10,
) -> BaseRetriever:
    """
    构建纯向量（稠密）检索器。

    Args:
        vector_store: FaissVectorStore 包装实例（含 .vector_store 与 .embeddings）
        k:            召回条数

    Returns:
        BaseRetriever
    """
    return vector_store.vector_store.as_retriever(
        search_kwargs={"k": k}
    )

# 通用组合器
def build_ensemble_retriever(
    bm25_retriever: BaseRetriever,
    vector_retriever: BaseRetriever,
    weights: Optional[List[float]] = None,
    c: int = DEFAULT_RRF_C,
) -> EnsembleRetriever:
    """
    通用组合器：把任意两个检索器融合为 EnsembleRetriever。

    - bm25_retriever:    BM25 路（build_bm25_retriever 产物）
    - vector_retriever:  向量路，可以是纯向量，也可以是 HyDE 检索器
                         （由工具层决定传哪种）
    - weights:           [BM25 权重, 向量权重]，默认 [0.4, 0.6]
    - c:                 RRF 平滑常数，默认 60
    """
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights or [DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT],
        c=c,
    )

# 便捷组合：BM25 + 纯向量
def build_base_retriever(
    vector_store: Any,
    all_chunks: List[Document],
    vector_k: int,
    bm25_k: int,
    use_bm25: bool = True,
    weights: Optional[List[float]] = None,
    c: int = DEFAULT_RRF_C,
) -> BaseRetriever:
    """
    便捷组合：BM25 + 纯向量。
    说明：
      - 不做深度决策，vector_k / bm25_k 由调用方传入。
      - use_bm25=False 或 all_chunks 为空 → 降级为纯向量检索。
      - 不涉及 HyDE；若要 HyDE，请用 build_vector_retriever 之外的路径
        （由工具层调 create_hyde_retriever + build_ensemble_retriever）。
    Returns:
        BaseRetriever
    """
    vector_retriever = build_vector_retriever(vector_store, k=vector_k)

    if not use_bm25 or not all_chunks:
        return vector_retriever

    bm25_retriever = build_bm25_retriever(all_chunks, k=bm25_k)
    return build_ensemble_retriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=weights,
        c=c,
    )

def build_filtered_vector_retriever(
    vector_store: Any,
    k: int = 10,
    category_filter: Optional[List[str]] = None,
    recall_multiplier: int = 3,
    min_results: int = 2,
) -> BaseRetriever:
    """
    构造一个"带分类元数据过滤"的向量检索器。
    
    实现方式：用一个轻量 Wrapper 包装 vector_store.search_with_filter，
    使其符合 LangChain BaseRetriever 接口。
    """
    from langchain_core.retrievers import BaseRetriever as _BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from pydantic import Field, ConfigDict

    class _FilteredRetriever(_BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        store: Any = Field(...)
        k: int = Field(default=10)
        categories: Optional[List[str]] = Field(default=None)
        recall_multiplier: int = Field(default=3)
        min_results: int = Field(default=2)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ):
            return self.store.search_with_filter(
                query=query,
                k=self.k,
                category_filter=self.categories,
                recall_multiplier=self.recall_multiplier,
                min_results=self.min_results,
            )

    return _FilteredRetriever(
        store=vector_store,
        k=k,
        categories=category_filter,
        recall_multiplier=recall_multiplier,
        min_results=min_results,
    )