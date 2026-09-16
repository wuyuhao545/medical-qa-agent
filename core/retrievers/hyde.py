# core/query_augmentation/hyde.py
"""
HyDE (Hypothetical Document Embeddings) 检索器
原理：使用 LLM 生成假设答案，再用假设答案的向量进行检索
优势：将口语化问题映射到文档风格的语义空间，提升召回质量
代价：每次查询多一次 LLM 调用

对外暴露：
    create_hyde_retriever(vector_store, embeddings, k=..., ...) -> HyDEVectorRetriever

注意：本模块只负责 HyDE 检索。
      "是否启用 HyDE" 由上层（depth_selector）根据档位决定，
      不在此处做 if use_hyde 判断。
"""

from typing import List, Optional, Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)
from pydantic import Field, ConfigDict

from models.llm import get_llm

# ===================== HyDE 默认配置 =====================
DEFAULT_HYDE_PROMPT = """你是一名资深医学专家。请针对下面的问题，以医学教科书的口吻写一段 100-200 字的假设性回答，覆盖可能的关键术语、病因、症状或处理原则。不需要精确到具体剂量，重点在术语和知识框架。
问题：{question}
假设性回答："""

DEFAULT_HYDE_K = 5
# ========================================================


class HyDEVectorRetriever(BaseRetriever):
    """
    使用 HyDE 增强的向量检索器。
    先由 LLM 生成假设答案，再对该答案进行向量检索。

    重要：`vector_store` 必须是底层向量库实例（含 similarity_search_by_vector），
          不是 FaissVectorStore 之类的包装类。

    推荐用 `create()` 类方法构造，避免直接覆盖 Pydantic 的 __init__。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Any = Field(description="底层向量库（需支持 similarity_search_by_vector）")
    embeddings: Any = Field(description="嵌入模型（需支持 embed_query）")
    llm: Any = Field(default=None, description="生成假设答案的 LLM")
    prompt: PromptTemplate = Field(
        default_factory=lambda: PromptTemplate.from_template(DEFAULT_HYDE_PROMPT)
    )
    k: int = Field(default=DEFAULT_HYDE_K, description="检索返回的文档数量")

    # ---------- 构造入口 ----------
    @classmethod
    def create(
        cls,
        vector_store,
        embeddings,
        llm=None,
        prompt_template: Optional[str] = None,
        k: int = DEFAULT_HYDE_K,
    ) -> "HyDEVectorRetriever":
        """推荐构造方式：内部完成 llm / prompt 的默认值填充与类型转换。"""
        return cls(
            vector_store=vector_store,
            embeddings=embeddings,
            llm=llm or get_llm(temperature=0),
            prompt=PromptTemplate.from_template(prompt_template or DEFAULT_HYDE_PROMPT),
            k=k,
        )

    # ---------- 内部工具 ----------
    def _gen_hypothetical(self, query: str) -> str:
        """生成假设答案；LLM 失败或返回空时，退回原 query（HyDE 只是增强，不应成为单点故障）。"""
        try:
            prompt_text = self.prompt.format(question=query)
            answer = self.llm.invoke(prompt_text).content
            answer = (answer or "").strip()
            return answer or query
        except Exception as e:
            print(f"⚠️ HyDE 生成失败，退回原 query: {e}")
            return query

    # ---------- 同步检索 ----------
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        hypothetical_answer = self._gen_hypothetical(query)
        query_vector = self.embeddings.embed_query(hypothetical_answer)
        return self.vector_store.similarity_search_by_vector(
            embedding=query_vector,
            k=self.k,
        )

    # ---------- 异步检索 ----------
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        try:
            prompt_text = self.prompt.format(question=query)
            resp = await self.llm.ainvoke(prompt_text)
            hypothetical_answer = (resp.content or "").strip() or query
        except Exception as e:
            print(f"⚠️ HyDE 异步生成失败，退回原 query: {e}")
            hypothetical_answer = query

        query_vector = self.embeddings.embed_query(hypothetical_answer)
        return self.vector_store.similarity_search_by_vector(
            embedding=query_vector,
            k=self.k,
        )


# ===================== 工厂函数（纯 HyDE，无分支） =====================
def create_hyde_retriever(
    vector_store,                       # 底层向量库实例（如 FAISS）
    embeddings,
    prompt_template: Optional[str] = None,
    k: int = DEFAULT_HYDE_K,
    llm=None,
) -> HyDEVectorRetriever:
    """
    创建一个 HyDE 检索器。始终返回 HyDEVectorRetriever。

    Args:
        vector_store:    底层向量库实例（需支持 similarity_search_by_vector）
        embeddings:      嵌入模型对象（需支持 embed_query）
        prompt_template: 自定义 HyDE 提示词模板（可选）
        k:               检索文档数量
        llm:             自定义 LLM；不传则用默认 get_llm(temperature=0)

    Returns:
        HyDEVectorRetriever 实例
    """
    return HyDEVectorRetriever.create(
        vector_store=vector_store,
        embeddings=embeddings,
        llm=llm,
        prompt_template=prompt_template,
        k=k,
    )