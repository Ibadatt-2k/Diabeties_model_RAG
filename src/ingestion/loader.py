from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    BSHTMLLoader,
    TextLoader,
)


_LOADERS = {
    ".pdf":  PyPDFLoader,
    ".html": BSHTMLLoader,
    ".htm":  BSHTMLLoader,
    ".md":   TextLoader,
    ".txt":  TextLoader,
}


def load_documents(docs_dir: str) -> List[Document]:
    """
    Recursively load all supported files from docs_dir.
    Attaches source_file and source_path metadata to every page/section.
    Supports: PDF, HTML, Markdown, plain text.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    documents: List[Document] = []

    for file_path in sorted(docs_path.rglob("*")):
        if not file_path.is_file():
            continue

        loader_class = _LOADERS.get(file_path.suffix.lower())
        if loader_class is None:
            continue  # unsupported extension — skip silently

        try:
            loader = loader_class(str(file_path))
            pages = loader.load()
            for doc in pages:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["source_path"] = str(file_path.relative_to(docs_path))
            documents.extend(pages)
            print(f"  Loaded: {file_path.name}  ({len(pages)} section(s))")
        except Exception as exc:
            print(f"  Warning: skipping {file_path.name} — {exc}")

    print(f"\nTotal sections loaded: {len(documents)}")
    return documents
