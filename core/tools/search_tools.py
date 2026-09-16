# tools/search_tools.py
"""
医学检索工具（Tool 层）。
职责：
  1. 用 LLM 对用户 query 判断检索深度 → "basic" / "standard" / "deep"
  2. 交给 depth_selector 按深度组装检索器
  3. 执行检索，把结果格式化为带出处的文本
  4. 注册成 LangChain StructuredTool，供 Agent 调用
本文件不实现检索算法，只做"调度 + 工具封装 + 结果格式化"。
熔断 + 降级链：
  主检索（按深度组装）→ 纯向量 → BM25 → 静态兜底
  每层独立熔断器，互不影响。
"""

import os
import sys

_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from typing import List, Optional, Dict

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import (
    FAISS_INDEX_DIR,
    ENABLE_CATEGORY_FILTER,
    CATEGORY_FILTER_RECALL_MULTIPLIER,
    CATEGORY_FILTER_MIN_RESULTS,
    CATEGORY_CLASSIFIER_TIMEOUT,
)

from utils.circuit_breaker import (
    get_breaker,
    CircuitOpenError,
    SlowCallCircuitBreaker,
)
from RAG.vector_stores.faiss_store import FaissVectorStore
from core.tools.deep_search import (
    get_depth_config,
    build_retriever_by_depth,
    DEPTH_BASIC,
    DEPTH_STANDARD,
    DEPTH_DEEP,
)


# ==================== 全局状态（由 main.py 注入） ====================
_ALL_CHUNKS: Optional[List[Document]] = None      # 小块（BM25 用）
_RAW_DOCUMENTS: Optional[List[Document]] = None   # 原始长文档（父检索用）
_vector_store: Optional[FaissVectorStore] = None
_classifier_llm = None


def set_all_chunks(chunks: List[Document]) -> None:
    global _ALL_CHUNKS
    _ALL_CHUNKS = chunks


def set_raw_documents(docs: List[Document]) -> None:
    global _RAW_DOCUMENTS
    _RAW_DOCUMENTS = docs


def get_vector_store() -> FaissVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FaissVectorStore(persist_dir=str(FAISS_INDEX_DIR))
        _vector_store.load_vector_store(verbose=False)
    return _vector_store


# ==================== LLM 深度分类器 ====================
_VALID_DEPTHS = {DEPTH_BASIC, DEPTH_STANDARD, DEPTH_DEEP}
_DEFAULT_DEPTH = DEPTH_STANDARD

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个医学问答系统的检索策略分析器。请阅读用户问题，"
        "判断应该使用哪种检索深度，并严格输出 JSON 对象，只包含一个字段：\n"
        '  "depth": 取值只能是 "basic" / "standard" / "deep"\n'
        "\n"
        "判定标准：\n"
        "  basic    —— 与医学无关的闲聊、常识性提问，或非常简单的医学名词解释。\n"
        "              例：'你好'、'今天天气如何'、'什么是血压'\n"
        "  standard —— 一般医学咨询：症状描述、常见病、用药常识、检查解读等。\n"
        "              例：'感冒了吃什么药'、'高血压饮食注意什么'\n"
        "  deep     —— 紧急/可能危及生命的症状，或需要深入、多角度的专业医学分析。\n"
        "              例：'突然胸痛喘不上气'、'大量咳血怎么办'、'持续高烧抽搐'\n"
        "\n"
        "只输出 JSON，不要输出任何其他文字或解释。",
    ),
    ("human", "{query}"),
])


def _get_classifier_llm():
    """懒加载分类用 LLM（走统一入口，自动挂 Token 统计）。"""
    global _classifier_llm
    if _classifier_llm is None:
        from models.llm import get_llm
        _classifier_llm = get_llm(temperature=0)
    return _classifier_llm


def _normalize_depth(label) -> str:
    """把 LLM 返回的深度标签清洗到合法值；认不出回退 standard。"""
    if not isinstance(label, str):
        return _DEFAULT_DEPTH
    label = label.strip().lower()
    return label if label in _VALID_DEPTHS else _DEFAULT_DEPTH


def classify_depth(query: str) -> str:
    """
    用 LLM 判断该 query 应使用的检索深度。
    返回 "basic" / "standard" / "deep"；失败或熔断时回退 "standard"。
    """
    cb = get_breaker(
        "depth_classifier",
        breaker_cls=SlowCallCircuitBreaker,
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        slow_call_threshold=8.0,   # 分类不该超过 8s
        slow_call_rate=0.5,
    )
    try:
        chain = CLASSIFIER_PROMPT | _get_classifier_llm() | JsonOutputParser()
        # 用 cb.call 包装 invoke，熔断器 OPEN 时快速失败
        result = cb.call(chain.invoke, {"query": query})
        return _normalize_depth(result.get("depth"))
    except CircuitOpenError as e:
        print(f"⚠️ 深度分类熔断，回退 standard: {e}")
        return _DEFAULT_DEPTH
    except Exception as e:
        print(f"⚠️ 深度分类失败，回退 standard: {e}")
        return _DEFAULT_DEPTH

# ==================== LLM 分类检测器 ====================
_CATEGORY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是医学知识分类器。请阅读用户问题，判断它最可能属于哪些医学分类。\n"
        "可多选，最多 2 个；如果无法判断，返回空列表。\n"
        "常见分类示例：内科、外科、妇产科、儿科、五官科、皮肤科、肿瘤科、"
        "中医、药学、急救、公共卫生、医学基础、未分类。\n\n"
        "严格输出 JSON，格式：{{\"categories\": [\"分类1\", \"分类2\"]}}\n"
        "不要输出任何其他文字。",
    ),
    ("human", "{query}"),
])

_category_classifier_llm = None


def _get_category_classifier_llm():
    global _category_classifier_llm
    if _category_classifier_llm is None:
        from models.llm import get_llm
        _category_classifier_llm = get_llm(temperature=0)
    return _category_classifier_llm


def classify_category(query: str) -> List[str]:
    """
    用 LLM 判断 query 相关的医学分类。
    失败/熔断/关闭时返回空列表（空列表 = 不过滤）。
    """
    if not ENABLE_CATEGORY_FILTER:
        return []

    cb = get_breaker(
        "category_classifier",
        breaker_cls=SlowCallCircuitBreaker,
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        slow_call_threshold=CATEGORY_CLASSIFIER_TIMEOUT,
        slow_call_rate=0.5,
    )
    try:
        chain = _CATEGORY_CLASSIFIER_PROMPT | _get_category_classifier_llm() | JsonOutputParser()
        result = cb.call(chain.invoke, {"query": query})
        cats = result.get("categories") if isinstance(result, dict) else None
        if not isinstance(cats, list):
            return []
        # 清洗：去空白、去重、限长
        cleaned = []
        for c in cats[:2]:
            s = str(c).strip()
            if s and s not in cleaned:
                cleaned.append(s)
        return cleaned
    except CircuitOpenError as e:
        print(f"⚠️ 分类检测熔断，跳过过滤: {e}")
        return []
    except Exception as e:
        print(f"⚠️ 分类检测失败，跳过过滤: {e}")
        return []

# ==================== 结果格式化 ====================
def _format_docs(docs: List[Document], max_len: int = 800) -> str:
    """把文档列表格式化为带出处的文本。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        if len(content) > max_len:
            content = content[:max_len] + "..."
        ref = f"【来源 {i}】{source}" + (f" (第{page}页)" if page else "")
        parts.append(f"{ref}\n{content}")
    return "\n\n---\n\n".join(parts)


# 核心检索（带熔断 + 三层降级
def _search_impl(query: str, top_k: int = 3) -> str:
    # ---------- 兼容 LLM 返回的异常格式 ----------
    if isinstance(query, dict):
        # 尝试从常见字段里取
        query = query.get("query") or query.get("text") or query.get("content") or str(query)
    if not isinstance(query, str):
        query = str(query)
    query = query.strip()
    if not query:
        return "未提供有效的查询内容。"
    depth = classify_depth(query) # 打印深度分类结果
    print(f"\n{'='*60}")
    print(f"[深度分类] query='{query}'")
    print(f"[深度分类] 判定结果 = {depth}")
    print(f"{'='*60}\n")

    config = get_depth_config(depth)
    effective_k = min(top_k, config["top_k"])
    """
    完整检索链路：
      query → LLM 判深度 → 组装检索器 → 执行 → 格式化

    熔断 + 降级：
      主检索 → 纯向量 → BM25 → 静态兜底
    每层独立熔断器，互不影响；每层失败/空结果都继续往下走。

    最终返回区分两种情况：
      - 有任一成功执行但无结果 → "未找到相关医学资料。"
      - 全部失败 / 熔断         → "检索服务暂时不可用，请稍后再试。"
    """
    # ---------- 0. 深度分类（内部已有熔断，失败回退 standard） ----------
    depth = classify_depth(query)
    config = get_depth_config(depth)
    effective_k = min(top_k, config["top_k"])

    # ---------- 1. 向量库 ----------
    vector_store = get_vector_store()
    if not vector_store.vector_store:
        return "知识库暂时不可用，请稍后再试。"

    # 记录“是否有任一层成功执行过”（哪怕返回空）
    any_layer_ran = False

    # ========== 0.5 分类检测（用于过滤；失败返回空列表）==========
    categories = classify_category(query)
    if categories:
        print(f"[分类过滤] 命中分类 = {categories}")
    else:
        print("[分类过滤] 未命中分类，走无过滤检索")

    # ========== 第一层：按深度组装的主检索 ==========
    main_cb = get_breaker(
        "main_retriever",
        failure_threshold=5,
        recovery_timeout=20.0,
    )
    try:
        retriever, _ = build_retriever_by_depth(
            depth=depth,
            vector_store=vector_store,
            all_chunks=_ALL_CHUNKS,
            raw_documents=_RAW_DOCUMENTS,
            category_filter=categories,   # ← 新增参数
        )
        docs = main_cb.call(retriever.invoke, query)[:effective_k]
        any_layer_ran = True
        if docs:
            tag = f"（检索深度={depth}"
            if categories:
                tag += f"，分类过滤={'/'.join(categories)}"
            tag += "）"
            return tag + "\n\n" + _format_docs(docs)
        print("ℹ️ 主检索返回空结果，尝试降级召回")
    except CircuitOpenError as e:
        print(f"⚠️ 主检索熔断，进入降级: {e}")
    except Exception as e:
        print(f"⚠️ 主检索失败，进入降级: {e}")

    # ========== 第一层备选：分类过滤向量检索（深度无过滤时才有意义）==========
    if categories and not _RAW_DOCUMENTS and ENABLE_CATEGORY_FILTER:
        try:
            from core.retrievers.base_retriever import build_filtered_vector_retriever
            filtered_retriever = build_filtered_vector_retriever(
                vector_store=vector_store,
                k=effective_k,
                category_filter=categories,
                recall_multiplier=CATEGORY_FILTER_RECALL_MULTIPLIER,
                min_results=CATEGORY_FILTER_MIN_RESULTS,
            )
            docs = main_cb.call(filtered_retriever.invoke, query)[:effective_k]
            if docs:
                any_layer_ran = True
                return (
                    f"（分类过滤向量检索：{'/'.join(categories)}）\n\n"
                    + _format_docs(docs)
                )
        except Exception as e:
            print(f"⚠️ 分类过滤向量检索失败: {e}")

# ========== 第二层：纯向量检索（独立熔断，带分类过滤）==========
    vec_cb = get_breaker(
        "faiss_search",
        failure_threshold=5,
        recovery_timeout=20.0,
    )
    try:
        # 优先走带过滤的检索；没分类时退化为普通检索
        if categories and ENABLE_CATEGORY_FILTER:
            docs = vec_cb.call(
                vector_store.search_with_filter,
                query,
                effective_k,
                categories,
                CATEGORY_FILTER_RECALL_MULTIPLIER,
                CATEGORY_FILTER_MIN_RESULTS,
            )[:effective_k]
        else:
            docs = vec_cb.call(
                vector_store.vector_store.similarity_search,
                query,
                k=effective_k,
            )[:effective_k]

        any_layer_ran = True
        if docs:
            tag = "（降级：向量检索"
            if categories:
                tag += f"，分类={'/'.join(categories)}"
            tag += "）"
            return tag + "\n\n" + _format_docs(docs)
        print("ℹ️ 向量降级返回空结果，尝试 BM25 降级")
    except CircuitOpenError as e:
        print(f"⚠️ 向量降级熔断: {e}")
    except Exception as e:
        print(f"⚠️ 向量降级失败: {e}")

    # ========== 第三层：BM25（独立熔断，构建 + 检索都受保护） ==========
    bm25_cb = get_breaker(
        "bm25_search",
        failure_threshold=5,
        recovery_timeout=20.0,
    )
    if not _ALL_CHUNKS:
        print("⚠️ BM25 降级跳过：_ALL_CHUNKS 为空，可能知识库未初始化")
    else:
        try:
            # 构建 + 检索整体交给熔断器，避免构建失败不计入失败计数
            def _bm25_search():
                from core.retrievers.base_retriever import build_bm25_retriever
                ret = build_bm25_retriever(_ALL_CHUNKS, k=effective_k)
                return ret.invoke(query)

            docs = bm25_cb.call(_bm25_search)[:effective_k]
            any_layer_ran = True
            if docs:
                return "（降级：BM25）\n\n" + _format_docs(docs)
            print("ℹ️ BM25 降级返回空结果")
        except CircuitOpenError as e:
            print(f"⚠️ BM25 降级熔断: {e}")
        except Exception as e:
            print(f"⚠️ BM25 降级失败: {e}")

    if any_layer_ran:                        # 兜底（区分“空结果”与“全失败”）
        return "未找到相关医学资料。"
    
    if docs:                                 # 返回引用来源
        result = f"（检索深度={depth}）\n\n" + _format_docs(docs)
        print(f"[DEBUG] 检索返回前 200 字: {result[:200]}")   # ← 加这行
        return result
    
    return "检索服务暂时不可用，请稍后再试。"


# ==================== 症状分析 ====================
def _analyze_impl(symptoms: str, duration: Optional[str] = None) -> str:
    if isinstance(symptoms, dict):
        symptoms = symptoms.get("symptoms") or symptoms.get("text") or str(symptoms)
    if not isinstance(symptoms, str):
        symptoms = str(symptoms)
    query = f"{symptoms} {'持续' + duration if duration else ''} 可能疾病分析"
    raw_result = _search_impl(query, top_k=3)   # 熔断已在内部生效，无需重复
    return (
        f"根据您描述的症状「{symptoms}」，检索到以下相关信息：\n\n"
        f"{raw_result}\n\n*注意：以上仅为知识检索结果，不构成诊断。*"
    )


# ==================== Pydantic Schema ====================
class SearchInput(BaseModel):
    query: str = Field(description="医学相关问题或症状描述")
    top_k: int = Field(default=3, description="返回的文档片段数量，最多不超过5")


class SymptomInput(BaseModel):
    symptoms: str = Field(description="用户描述的症状")
    duration: Optional[str] = Field(default=None, description="症状持续时间")


# ==================== 工具注册 ====================
search_medical_tool = StructuredTool.from_function(
    func=_search_impl,
    name="search_medical_knowledge",
    description="搜索权威医学知识库，返回带出处的相关医学知识片段。",
    args_schema=SearchInput,
    return_direct=False,
)

analyze_symptoms_tool = StructuredTool.from_function(
    func=_analyze_impl,
    name="analyze_symptoms",
    description="根据用户描述的症状组合，检索可能相关的疾病知识。",
    args_schema=SymptomInput,
)

ALL_TOOLS = [search_medical_tool, analyze_symptoms_tool]