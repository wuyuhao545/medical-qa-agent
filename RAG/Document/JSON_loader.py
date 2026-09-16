# core/loaders/json_document.py
"""
JSONL 加载器：逐行解析，规范化 categories，title 拼接到正文前。
"""
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document


JSON_EXTENSIONS = (".jsonl",)


def _normalize_categories(raw) -> List[str]:
    """把 categories 规范化为非空的 List[str]。"""
    if isinstance(raw, list):
        cats = [str(c).strip() for c in raw if str(c).strip()]
    elif isinstance(raw, str) and raw.strip():
        cats = [raw.strip()]
    else:
        cats = []
    # 没有分类打个默认标签，方便过滤兜底
    return cats or ["未分类"]


def _build_doc(data: dict, file_path: Path, line_num: int) -> Document | None:
    """从一行 JSON 构造 Document，内容为空则返回 None。"""
    content = data.get("content") or data.get("text") or data.get("body") or ""
    if not content or not content.strip():
        return None

    categories = _normalize_categories(data.get("categories"))
    title = str(data.get("title", "")).strip()

    # 把 title 拼到正文前，提升向量/BM25 命中率
    if title and title not in content[:50]:
        page_content = f"【{title}】\n{content}"
    else:
        page_content = content

    metadata = {
        "source": str(file_path),
        "title": title,
        "url": str(data.get("url", "")),
        "categories": categories,
        "categories_str": "|".join(categories),   # 兼容简单过滤器
        "content_length": int(data.get("content_length", len(content))),
        "language": str(data.get("language", "zh-CN")),
        "line": line_num,
    }
    return Document(page_content=page_content, metadata=metadata)


def load_jsonl_file(path: Path, verbose: bool = True) -> List[Document]:
    """加载单个 .jsonl 文件。"""
    docs: List[Document] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # 只在每 1000 行报一次，避免刷屏
                    if verbose and line_num % 1000 == 0:
                        print(f"  ⚠️ {path.name} 第 {line_num} 行 JSON 解析失败，跳过")
                    continue

                doc = _build_doc(data, path, line_num)
                if doc is not None:
                    docs.append(doc)

        if verbose:
            print(f"  ✅ {path.name} -> {len(docs)} 个文档")
    except Exception as e:
        if verbose:
            print(f"  ❌ {path.name} 加载失败: {e}")
    return docs


def load_jsonl_dir(data_dir: Path, verbose: bool = True) -> List[Document]:
    """递归加载目录下所有 .jsonl 文件。"""
    all_docs: List[Document] = []
    for fp in data_dir.rglob("*.jsonl"):
        if fp.is_file():
            all_docs.extend(load_jsonl_file(fp, verbose=verbose))
    return all_docs