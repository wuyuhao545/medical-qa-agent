# core/vector_stores/faiss_store.py
import os
from typing import List,Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import FAISS_INDEX_DIR
from RAG.embedding_factory import get_embeddings


class FaissVectorStore:
    def __init__(self, persist_dir: str = None, embeddings=None):
        self.persist_dir = persist_dir or str(FAISS_INDEX_DIR)
        self.embeddings = embeddings or get_embeddings()
        self.vector_store = None

    def build_vector_store(self, documents: List[Document], verbose: bool = True) -> bool:
        if not documents:
            if verbose:
                print("⚠️ 没有文档可构建向量库")
            return False
        if verbose:
            print("📝 正在使用 FAISS 生成向量索引...")
        self.vector_store = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_store.save_local(self.persist_dir)
        if verbose:
            print(f"💾 FAISS 向量库已保存到: {self.persist_dir}")
        return True

    def load_vector_store(self, verbose: bool = True) -> bool:
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
            if verbose:
                print(f"✅ 加载 FAISS 向量库: {self.persist_dir}")
            return True
        if verbose:
            print(f"⚠️ FAISS 向量库不存在: {self.persist_dir}")
        return False

    def add_documents(self, documents: List[Document], verbose: bool = True) -> bool:
        if not documents:
            return False
        # 修正：加载失败时不静默重建，避免覆盖旧库
        if self.vector_store is None:
            if not self.load_vector_store(verbose=False):
                if verbose:
                    print("❌ 向量库不存在，增量添加中止（不重建以避免丢数据）")
                return False
        try:
            self.vector_store.add_documents(documents)
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vector_store.save_local(self.persist_dir)
            if verbose:
                print(f"✅ 增量添加 {len(documents)} 个文档到 FAISS")
            return True
        except Exception as e:
            if verbose:
                print(f"❌ 增量添加失败: {e}")
            return False

    def get_retriever(self, k: int = 3):
        if not self.vector_store:
            raise ValueError("向量库未初始化")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4):
        if not self.vector_store:
            if not self.load_vector_store(verbose=False):
                raise ValueError("向量库未初始化且无法加载。")
        return self.vector_store.similarity_search_by_vector(embedding, k=k)

    def search_with_filter(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[List[str]] = None,
        recall_multiplier: int = 3,
        min_results: int = 2,
    ) -> List[Document]:
        """
        基于分类元数据的过滤检索。
        
        策略：
          1. 先无过滤召回 k * recall_multiplier 条
          2. 按 category_filter 过滤
          3. 过滤后 < min_results → 自动放宽（返回不过滤的 top_k）
          4. 过滤后 >= min_results → 返回过滤后 top_k
        
        Args:
            query:              查询文本
            k:                  最终返回条数
            category_filter:    需要匹配的分类列表；None / 空表示不过滤
            recall_multiplier:  先多召回的倍数
            min_results:        过滤后少于此值就放宽
        """
        if not self.vector_store:
            if not self.load_vector_store(verbose=False):
                raise ValueError("向量库未初始化且无法加载")

        # 不需要过滤 → 直接走原生检索
        if not category_filter:
            return self.vector_store.similarity_search(query, k=k)

        recall_k = max(k * recall_multiplier, k)

        # 自定义过滤函数：doc.metadata['categories'] 与目标分类有交集即保留
        target = set(category_filter)

        # ✅ 修正：FAISS 的 filter 接收的是 metadata 字典，不是 Document
        target = set(category_filter)

        def _filter(metadata: dict) -> bool:
            """FAISS 会传 doc.metadata 进来（dict），返回 True 保留"""
            if not isinstance(metadata, dict):
                return False
            cats = metadata.get("categories") or []
            if isinstance(cats, str):
                cats = [cats]
            try:
                return bool(target & set(cats))
            except TypeError:
                return False

        try:
            docs_filtered = self.vector_store.similarity_search(
                query, k=recall_k, filter=_filter
            )[:k]
        except Exception as e:
            print(f"⚠️ 分类过滤失败，退回无过滤检索: {e}")
            docs_filtered = []

        # 过滤结果太少 → 放宽（返回无过滤的 top_k）
        if len(docs_filtered) < min_results:
            print(f"ℹ️ 分类过滤后仅 {len(docs_filtered)} 条，放宽过滤")
            return self.vector_store.similarity_search(query, k=k)

        return docs_filtered