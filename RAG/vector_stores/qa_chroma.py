# core/vector_stores/qa_chroma.py
"""
QA 问答记录存储（基于 Chroma）。

- 用于持久化用户问答对，便于后续检索相似历史问答。
- Chroma 0.4+ 起自动持久化，无需手动调用 persist()。
"""

import os
from typing import Optional, List, Dict, Any

from langchain_chroma import Chroma                       # ← 新版包（推荐）
from langchain_core.documents import Document

from config import CHROMA_QA_DIR
from RAG.embedding_factory import get_embeddings


class QaVectorStore:
    def __init__(self, persist_dir: str = None, embeddings=None):
        self.persist_dir = persist_dir or str(CHROMA_QA_DIR)
        self.embeddings = embeddings or get_embeddings()
        self.vector_store: Optional[Chroma] = None

    # ---------- 加载 / 创建 ----------
    def load_or_create(self, verbose: bool = True) -> bool:
        """加载已有 Chroma QA 库；不存在则创建一个空库。"""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            if verbose:
                print(f"✅ 加载 Chroma QA 库: {self.persist_dir}")
        else:
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            # 注：Chroma 0.4+ 自动持久化，无需再调 persist()
            if verbose:
                print(f"🆕 创建新的 Chroma QA 库: {self.persist_dir}")
        return self.vector_store is not None

    # ---------- 插入一条问答记录 ----------
    def add_qa_pair(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """插入一条问答记录（自动持久化）。"""
        if self.vector_store is None:
            self.load_or_create(verbose=False)

        doc = Document(
            page_content=f"问：{question}\n答：{answer}",
            metadata=metadata or {},
        )
        self.vector_store.add_documents([doc])
        print(f"📝 QA 记录已保存")

    # ---------- 检索相似历史问答 ----------
    def retrieve_similar_qa(self, query: str, k: int = 3) -> List[Document]:
        """根据当前问题检索相似的 QA 记录。"""
        if self.vector_store is None:
            self.load_or_create(verbose=False)
        return self.vector_store.similarity_search(query, k=k)