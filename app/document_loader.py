from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(path: str | Path) -> list[Document]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()

    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8").load()

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}")


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        documents.extend(load_document(path))
    return documents
