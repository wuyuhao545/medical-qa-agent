# core/retrievers/parent_doc_retriever.py
"""
父文档检索器（独立 FAISS 版本）：
- 索引：长文档 → 父块(大) 进 docstore；子块(小) 进独立 FAISS
- 检索：子块 ANN 召回 → 返回完整父块
- 与主知识库 FAISS 完全隔离，互不污染

对外暴露：
  - build_parent_retriever(all_documents, ...)     构建（首次 / 强制重建）
  - get_parent_retriever(all_documents, ...)       带缓存的获取
  - add_documents_to_parent_retriever(new_docs)    增量更新
  - clear_parent_retriever_cache()                 清缓存（文件监控用）
"""

import shutil
import uuid
from pathlib import Path
from typing import List, Any, Optional

from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import (
    PARENT_DOCSTORE_DIR,
    PARENT_FAISS_DIR,
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    PARENT_OVERLAP,
    PARENT_CHUNK_SIZE,
    RETRIEVAL_DEPTH_CONFIG,
)
from RAG.embedding_factory import get_embeddings

SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
_PARENT_RETRIEVER_CACHE: dict = {}
_INIT_MARKER = "__parent_retriever_init__"


# ==================== 内部工具 ====================

def _load_or_init_faiss(faiss_path: Path, embeddings) -> Optional[FAISS]:
    """
    加载已有 FAISS；不存在 / 损坏 / 为空 时返回 None。
    """
    if not faiss_path.exists() or not any(faiss_path.iterdir()):
        return None
    try:
        return FAISS.load_local(
            str(faiss_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"⚠️ 加载父文档 FAISS 失败: {e}")
        return None


def _create_empty_faiss(embeddings) -> FAISS:
    """
    FAISS 不支持真正的空初始化，用一条占位文档兜底。
    占位文档内容是无意义字符串，永远不会命中真实查询。
    """
    init_doc = Document(page_content=_INIT_MARKER, metadata={"_init": True})
    return FAISS.from_documents([init_doc], embeddings)


def _clear_dirs(*dirs: Path):
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


# ==================== 构建 ====================

def build_parent_retriever(
    all_documents: List[Document],
    search_k: Optional[int] = None,
    verbose: bool = True,
    rebuild: bool = False,
) -> Optional[ParentDocumentRetriever]:
    """
    构建父文档检索器（独立 FAISS）。
    Args:
        all_documents: 原始长文档列表
        search_k:      检索时召回的父块数，默认取 deep 档 top_k
        verbose:       是否打印构建日志
        rebuild:       是否强制重建（清空 docstore + FAISS）
    Returns:
        ParentDocumentRetriever；如果既没有索引又没有原始文档，返回 None
    """
    if search_k is None:
        search_k = RETRIEVAL_DEPTH_CONFIG["deep"]["top_k"]

    embeddings = get_embeddings()
    docstore_path = Path(PARENT_DOCSTORE_DIR)
    faiss_path = Path(PARENT_FAISS_DIR)

    # ---------- 强制重建 ----------
    if rebuild:
        _clear_dirs(docstore_path, faiss_path)
        if verbose:
            print(f"🧹 已清空旧父文档索引（{docstore_path} / {faiss_path}）")
    else:
        docstore_path.mkdir(parents=True, exist_ok=True)
        faiss_path.mkdir(parents=True, exist_ok=True)

    # ---------- 一致性检查 ----------
    docstore_empty = not any(docstore_path.iterdir())
    faiss_vectorstore = _load_or_init_faiss(faiss_path, embeddings)
    faiss_missing = faiss_vectorstore is None

    # 两边状态不一致 → 一起重建
    if docstore_empty != faiss_missing:
        if verbose:
            print("⚠️ 父文档索引不一致（docstore / FAISS 只存在其一），将一起重建")
        _clear_dirs(docstore_path, faiss_path)
        docstore_empty = True
        faiss_vectorstore = None

    # 新代码
    file_store = LocalFileStore(root_path=str(docstore_path))
    docstore = create_kv_docstore(file_store)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_OVERLAP,
        separators=SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        separators=SEPARATORS,
    )

    # ---------- 决定是否需要写数据 ----------
    need_build = rebuild or docstore_empty or (faiss_vectorstore is None)

    # 如果 FAISS 不存在，先给它一个空壳
    if faiss_vectorstore is None:
        faiss_vectorstore = _create_empty_faiss(embeddings)
        if verbose:
            print(f"🆕 已创建空的父文档 FAISS 索引: {faiss_path}")

    retriever = ParentDocumentRetriever(
        vectorstore=faiss_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": search_k},
    )

    # ---------- 写入文档 ----------
    if need_build:
        if not all_documents:
            if verbose:
                print("⚠️ 未传入原始文档，跳过父文档索引构建")
        else:
            action = "重建" if rebuild else "首次构建"
            if verbose:
                print(f"📥 正在{action}父文档索引（父块 → {docstore_path}，子块 → {faiss_path}）...")
            retriever.add_documents(all_documents, ids=None)
            faiss_vectorstore.save_local(str(faiss_path))
            if verbose:
                print("✅ 父文档索引构建完成")
    else:
        if verbose:
            print(f"✅ 复用已有父文档索引（docstore={docstore_path}, faiss={faiss_path}）")

    return retriever


# ==================== 带缓存获取 ====================

def get_parent_retriever(
    all_documents: List[Document],
    search_k: int,
    rebuild: bool = False,
    verbose: bool = False,
) -> Optional[ParentDocumentRetriever]:
    """
    带缓存的获取接口。
    - 首次调用 → 真正构建（写子块进独立 FAISS、父块进 LocalFileStore）
    - 之后复用，仅动态更新 search_kwargs
    """
    if "parent" not in _PARENT_RETRIEVER_CACHE or rebuild:
        retriever = build_parent_retriever(
            all_documents=all_documents,
            search_k=search_k,
            rebuild=rebuild,
            verbose=verbose,
        )
        if retriever is None:
            return None
        _PARENT_RETRIEVER_CACHE["parent"] = retriever

    _PARENT_RETRIEVER_CACHE["parent"].search_kwargs = {"k": search_k}
    return _PARENT_RETRIEVER_CACHE["parent"]


# ==================== 增量更新 ====================

def add_documents_to_parent_retriever(
    new_documents: List[Document],
    verbose: bool = True,
) -> bool:
    """
    向已有的父文档索引增量添加文档。
    要求父检索器已经在缓存中（即至少构建过一次）。
    """
    if not new_documents:
        return False

    retriever = _PARENT_RETRIEVER_CACHE.get("parent")
    if retriever is None:
        if verbose:
            print("⚠️ 父文档检索器尚未构建，跳过增量更新")
        return False

    try:
        retriever.add_documents(new_documents, ids=None)
        # 持久化独立 FAISS
        faiss_path = Path(PARENT_FAISS_DIR)
        retriever.vectorstore.save_local(str(faiss_path))
        if verbose:
            print(f"✅ 增量添加 {len(new_documents)} 个文档到父文档索引")
        return True
    except Exception as e:
        if verbose:
            print(f"❌ 父文档索引增量添加失败: {e}")
        return False


# ==================== 缓存清理 ====================

def clear_parent_retriever_cache():
    """清空父检索器缓存（文件监控触发时调用，下次会重新构建）。"""
    _PARENT_RETRIEVER_CACHE.clear()