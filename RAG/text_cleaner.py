from typing import List
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """移除无效代理字符，返回合法 UTF-8 字符串。"""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def clean_documents(docs: List[Document]) -> List[Document]:
    """
    批量清洗 Document 列表：
      - 清洗 page_content
      - 丢弃清洗后为空的文档
    """
    cleaned = []
    for doc in docs:
        if not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)
        doc.page_content = clean_text(doc.page_content)
        if doc.page_content.strip():
            cleaned.append(doc)
    return cleaned