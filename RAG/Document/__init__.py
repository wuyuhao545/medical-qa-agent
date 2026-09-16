# core/loaders/__init__.py
from RAG.Document.document_loader import (
    load_documents,
    load_documents_from_paths,
)

__all__ = ["load_documents", "load_documents_from_paths"]