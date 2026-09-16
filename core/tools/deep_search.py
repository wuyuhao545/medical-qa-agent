# core/tools/depth_selector.py
"""
检索深度组装器。
职责：
  给定 depth（"basic" / "standard" / "deep"），
  先按一级深度取出该档位的二级参数，
  再按二级参数决定启用哪些检索器，组装成完整检索器。
两级结构：
  一级（depth）     → 决定用哪一套参数
  二级（子参数）     → use_hyde / use_bm25 / use_reranker / use_parent / top_k 决定用哪些检索器、返回多少条
"""

from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import RETRIEVAL_DEPTH_CONFIG, USE_PARENT_RETRIEVER
from core.retrievers.base_retriever import (
    build_bm25_retriever,
    build_vector_retriever,
    build_ensemble_retriever,
)
from core.retrievers.rerank import wrap_with_reranker, RERANKER_MODEL
from core.retrievers.parent_doc_retriever import get_parent_retriever

# 一级：深度常量
DEPTH_BASIC = "basic"
DEPTH_STANDARD = "standard"
DEPTH_DEEP = "deep"
_VALID_DEPTHS = {DEPTH_BASIC, DEPTH_STANDARD, DEPTH_DEEP}

#  一级：取二级参数
def get_depth_config(depth: str) -> dict:
    """
    一级对比：把 depth 归一化后，取出对应的二级参数集合。
    非法 / 未知深度 → 回退 standard。
    """
    if depth not in _VALID_DEPTHS:
        depth = DEPTH_STANDARD
    return RETRIEVAL_DEPTH_CONFIG[depth]


# 二级：按参数组装"传统多路召回" 
def _build_traditional_retriever(
    vector_store: Any,
    all_chunks: Optional[List[Document]],
    recall_k: int,
    use_hyde: bool,
    use_bm25: bool,
    category_filter: Optional[List[str]] = None,   # ← 新增
) -> BaseRetriever:
    """
    二级组装（传统多路召回）：
      - 向量路：use_hyde ? HyDE 检索器 : 纯向量（可带分类过滤）
      - BM25 路：use_bm25 ? 启用 : 跳过
      - 两路都启用 → Ensemble 融合；只启用一路 → 直接返回该路
    """
    if use_hyde:
        from core.retrievers.hyde import create_hyde_retriever
        vector_retriever = create_hyde_retriever(
            vector_store=vector_store.vector_store,
            embeddings=vector_store.embeddings,
            k=recall_k,
        )
    else:
        # 无 HyDE 时，如果开了分类过滤，走带过滤的向量检索器
        if category_filter:
            from core.retrievers.base_retriever import build_filtered_vector_retriever
            from config import (
                CATEGORY_FILTER_RECALL_MULTIPLIER,
                CATEGORY_FILTER_MIN_RESULTS,
            )
            vector_retriever = build_filtered_vector_retriever(
                vector_store=vector_store,
                k=recall_k,
                category_filter=category_filter,
                recall_multiplier=CATEGORY_FILTER_RECALL_MULTIPLIER,
                min_results=CATEGORY_FILTER_MIN_RESULTS,
            )
        else:
            vector_retriever = build_vector_retriever(vector_store, k=recall_k)

    if not use_bm25 or not all_chunks:
        return vector_retriever

    bm25_retriever = build_bm25_retriever(all_chunks, k=recall_k)
    return build_ensemble_retriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
    )

# 主入口：按深度组装完整检索器 
def build_retriever_by_depth(
    depth: str,
    vector_store: Any,
    all_chunks: Optional[List[Document]] = None,
    raw_documents: Optional[List[Document]] = None,
    category_filter: Optional[List[str]] = None,   # ← 新增
) -> Tuple[BaseRetriever, int]:
    """
    按深度组装检索器，返回 (retriever, top_k)。
    """
    config = get_depth_config(depth)

    top_k        = config["top_k"]
    use_hyde     = config.get("use_hyde", False)
    use_bm25     = config.get("use_bm25", False)
    use_reranker = config.get("use_reranker", False)
    use_parent   = config.get("use_parent", False)

    recall_k = top_k + 2
    use_parent_now = USE_PARENT_RETRIEVER and use_parent and bool(raw_documents)

    if use_parent_now:
        base_retriever = get_parent_retriever(
            all_documents=raw_documents,
            search_k=recall_k,
            verbose=False,
        )
        if base_retriever is None:
            base_retriever = _build_traditional_retriever(
                vector_store=vector_store,
                all_chunks=all_chunks,
                recall_k=recall_k,
                use_hyde=use_hyde,
                use_bm25=use_bm25,
                category_filter=category_filter,   # ← 传递
            )
    else:
        base_retriever = _build_traditional_retriever(
            vector_store=vector_store,
            all_chunks=all_chunks,
            recall_k=recall_k,
            use_hyde=use_hyde,
            use_bm25=use_bm25,
            category_filter=category_filter,       # ← 传递
        )

    if use_reranker:
        retriever = wrap_with_reranker(
            base_retriever=base_retriever,
            model_name=RERANKER_MODEL,
            top_n=top_k,
        )
    else:
        retriever = base_retriever

    return retriever, top_k