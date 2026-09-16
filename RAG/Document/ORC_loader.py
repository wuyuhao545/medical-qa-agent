# core/loaders/ocr_document.py
"""
OCR 加载器：PDF + 图片（jpg/jpeg/png/bmp/tiff）。
优先使用 Unstructured（支持扫描件），缺依赖时 PDF 退回 PyPDFLoader。
"""
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import (
        UnstructuredPDFLoader,
        UnstructuredImageLoader,
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


PDF_EXTENSIONS = (".pdf",)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
OCR_EXTENSIONS = PDF_EXTENSIONS + IMAGE_EXTENSIONS

# OCR 语言：中英文
_OCR_LANGUAGES = ["chi_sim", "eng"]


def _load_pdf(path: Path, verbose: bool) -> List[Document]:
    if UNSTRUCTURED_AVAILABLE:
        loader = UnstructuredPDFLoader(
            str(path),
            strategy="auto",
            mode="single",
            languages=_OCR_LANGUAGES,
        )
    else:
        loader = PyPDFLoader(str(path))
    return loader.load()


def _load_image(path: Path, verbose: bool) -> List[Document]:
    if not UNSTRUCTURED_AVAILABLE:
        if verbose:
            print(f"  ⚠️ 缺少 unstructured，跳过图片 {path.name}")
        return []
    loader = UnstructuredImageLoader(
        str(path),
        strategy="auto",
        mode="single",
        languages=_OCR_LANGUAGES,
    )
    return loader.load()


def load_ocr_file(path: Path, verbose: bool = True) -> List[Document]:
    """加载单个 PDF 或图片。"""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            docs = _load_pdf(path, verbose)
        elif ext in IMAGE_EXTENSIONS:
            docs = _load_image(path, verbose)
        else:
            if verbose:
                print(f"  ⚠️ 不是 OCR 格式: {ext}")
            return []

        if verbose:
            print(f"  ✅ {path.name} -> {len(docs)} 个片段")
        return docs
    except Exception as e:
        if verbose:
            print(f"  ❌ {path.name} 加载失败: {e}")
        return []


def load_ocr_dir(data_dir: Path, verbose: bool = True) -> List[Document]:
    """递归加载目录下所有 PDF 和图片。"""
    if verbose:
        if UNSTRUCTURED_AVAILABLE:
            print("  ✅ Unstructured 已加载，支持 PDF 和图片 OCR")
        else:
            print("  ⚠️ 未安装 unstructured，PDF 退回 PyPDFLoader，图片无法处理")

    all_docs: List[Document] = []
    for ext in OCR_EXTENSIONS:
        for fp in data_dir.rglob(f"*{ext}"):
            if fp.is_file():
                all_docs.extend(load_ocr_file(fp, verbose=verbose))
    return all_docs