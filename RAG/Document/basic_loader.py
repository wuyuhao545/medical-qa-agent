# core/loaders/basic_document.py
"""
基础格式加载器：.txt / .md / .markdown / .docx / .doc / .csv
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)
from langchain_core.documents import Document


# 扩展名 -> loader 配置
_BASIC_CONFIG = {
    ".txt":      {"cls": TextLoader, "kwargs": {"encoding": "utf-8"}},
    ".md":       {"cls": TextLoader, "kwargs": {"encoding": "utf-8"}},
    ".markdown": {"cls": TextLoader, "kwargs": {"encoding": "utf-8"}},
    ".docx":     {"cls": Docx2txtLoader, "kwargs": {}},
    ".doc":      {"cls": UnstructuredWordDocumentLoader, "kwargs": {}},
    ".csv":      {"cls": CSVLoader, "kwargs": {}},
}

BASIC_EXTENSIONS = tuple(_BASIC_CONFIG.keys())


def load_basic_file(path: Path, verbose: bool = True) -> List[Document]:
    """加载单个基础格式文件。"""
    ext = path.suffix.lower()
    if ext not in _BASIC_CONFIG:
        if verbose:
            print(f"  ⚠️ 不是基础格式: {ext}")
        return []

    cfg = _BASIC_CONFIG[ext]
    try:
        loader = cfg["cls"](str(path), **cfg["kwargs"])
        docs = loader.load()
        if verbose:
            print(f"  ✅ {path.name} -> {len(docs)} 个片段")
        return docs
    except Exception as e:
        if verbose:
            print(f"  ❌ {path.name} 加载失败: {e}")
        return []


def load_basic_dir(data_dir: Path, verbose: bool = True) -> List[Document]:
    """递归加载目录下所有基础格式文件。"""
    all_docs: List[Document] = []
    for ext in BASIC_EXTENSIONS:
        for fp in data_dir.rglob(f"*{ext}"):
            if fp.is_file():
                all_docs.extend(load_basic_file(fp, verbose=verbose))
    return all_docs