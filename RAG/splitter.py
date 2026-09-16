# core/splitters/splitter_manager.py
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
from RAG.embedding_factory import get_embeddings

class SplitterManager:
    """
    文本切分器管理器：根据策略返回不同的切分器实例
    """
    def __init__(
        self,
        strategy: str = "recursive",   # "recursive", "semantic", "markdown"
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        self._embeddings = None
        # 在 __init__ 里
        if strategy == "semantic":
            self._embeddings = get_embeddings()

    def get_splitter(self) -> Any:
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
        elif self.strategy == "semantic":
            return SemanticChunker(
                embeddings=self._embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.95,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.strategy == "markdown":
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header1"),
                    ("##", "Header2"),
                    ("###", "Header3"),
                ]
            )
        else:
            raise ValueError(f"不支持的切分策略: {self.strategy}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if self.strategy == "markdown":
            md_splitter = self.get_splitter()
            all_chunks = []
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            for doc in documents:
                if doc.metadata.get("source", "").endswith((".md", ".markdown")):
                    header_splits = md_splitter.split_text(doc.page_content)
                    for header_doc in header_splits:
                        sub_chunks = recursive_splitter.split_documents([header_doc])
                        all_chunks.extend(sub_chunks)
                else:
                    chunks = recursive_splitter.split_documents([doc])
                    all_chunks.extend(chunks)
            return all_chunks
        elif self.strategy == "semantic":
            splitter = self.get_splitter()
            return splitter.split_documents(documents)
        else:  # recursive
            splitter = self.get_splitter()
            return splitter.split_documents(documents)