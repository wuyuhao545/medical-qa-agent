# core/loaders/document_loader.py
"""
文档加载统一入口。

职责：
  1. 调用三类加载器：基础格式 / JSONL / OCR
  2. 在唯一位置做清洗（clean_text）
  3. 对外暴露 load_documents / load_documents_from_paths

清洗点说明：
  - 清洗放在"加载后、分块前"，保证 raw_documents 和 chunks 都干净
  - 父文档检索器拿到的原始文档也继承干净文本
  - 检索层（search_tools）不再重复清洗
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from .basic_loader import (load_basic_dir, load_basic_file, BASIC_EXTENSIONS)
from .JSON_loader import (load_jsonl_dir, load_jsonl_file, JSON_EXTENSIONS)
from .ORC_loader import (load_ocr_dir, load_ocr_file, OCR_EXTENSIONS)
# 清洗工具
try:
    from RAG.text_cleaner import clean_documents
except ImportError:
    # 兜底：内联实现（避免路径问题导致崩）
    def clean_documents(docs: List[Document]) -> List[Document]:
        cleaned = []
        for doc in docs:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)
            doc.page_content = (
                doc.page_content
                .encode("utf-8", errors="ignore")
                .decode("utf-8")
            )
            if doc.page_content.strip():
                cleaned.append(doc)
        return cleaned

# 目录扫描 
def load_documents(
    data_dir: Optional[str] = None,
    verbose: bool = True,
    clean: bool = True,
) -> List[Document]:
    """
    加载目录下所有支持格式的文档。
    返回：清洗后的 Document 列表（可直接传给 SplitterManager）。
    """
    if data_dir is None:
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        data_dir = str(project_root / "data" / "medical_knowledge")

    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"目录不存在: {data_dir}")

    data_dir_path = Path(data_dir)
    if verbose:
        print(f"📂 扫描目录: {data_dir_path}")

    all_docs: List[Document] = []

    # 1. 基础格式
    if verbose:
        print(f"🔍 [基础格式] {BASIC_EXTENSIONS}")
    basic_docs = load_basic_dir(data_dir_path, verbose=verbose)
    all_docs.extend(basic_docs)
    if verbose:
        print(f"   小计: {len(basic_docs)} 个片段")

    # 2. JSONL
    if verbose:
        print(f"🔍 [JSONL] {JSON_EXTENSIONS}")
    json_docs = load_jsonl_dir(data_dir_path, verbose=verbose)
    all_docs.extend(json_docs)
    if verbose:
        print(f"   小计: {len(json_docs)} 个片段")

    # 3. OCR（PDF + 图片）
    if verbose:
        print(f"🔍 [OCR] {OCR_EXTENSIONS}")
    ocr_docs = load_ocr_dir(data_dir_path, verbose=verbose)
    all_docs.extend(ocr_docs)
    if verbose:
        print(f"   小计: {len(ocr_docs)} 个片段")

    if verbose:
        print(f"📦 总计加载 {len(all_docs)} 个文档片段")

    # ---------- 唯一清洗点 ----------
    if clean:
        all_docs = clean_documents(all_docs)
        if verbose:
            print(f"🧹 清洗完成，剩余 {len(all_docs)} 个有效文档")

    return all_docs


# ==================== 按路径加载（增量更新用） ====================

# 三类扩展名映射
_BASIC_EXTS = set(BASIC_EXTENSIONS)
_JSON_EXTS = set(JSON_EXTENSIONS)
_OCR_EXTS = set(OCR_EXTENSIONS)


def load_documents_from_paths(
    file_paths: List[Path],
    verbose: bool = True,
    clean: bool = True,
) -> List[Document]:
    """
    按指定路径加载（用于增量更新）。
    根据扩展名分派给对应加载器。
    """
    all_docs: List[Document] = []

    for fp in file_paths:
        if not fp.exists():
            if verbose:
                print(f"  ⚠️ 文件不存在: {fp}")
            continue

        ext = fp.suffix.lower()

        if ext in _BASIC_EXTS:
            docs = load_basic_file(fp, verbose=verbose)
        elif ext in _JSON_EXTS:
            docs = load_jsonl_file(fp, verbose=verbose)
        elif ext in _OCR_EXTS:
            docs = load_ocr_file(fp, verbose=verbose)
        else:
            if verbose:
                print(f"  ⚠️ 未知格式: {ext}，跳过 {fp.name}")
            continue

        all_docs.extend(docs)

    # ---------- 唯一清洗点 ----------
    if clean:
        all_docs = clean_documents(all_docs)
        if verbose:
            print(f"🧹 清洗完成，剩余 {len(all_docs)} 个有效文档")

    return all_docs