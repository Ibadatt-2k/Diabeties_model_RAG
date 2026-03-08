from typing import List

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into token-counted chunks using tiktoken.
    Adds chunk_index and chunk_id metadata to every chunk.

    Settings read from config/settings.yaml:
      chunking.chunk_size    — target tokens per chunk   (default 750)
      chunking.chunk_overlap — overlap tokens            (default 100)
    """
    settings = _load_settings()
    chunk_size    = settings["chunking"]["chunk_size"]
    chunk_overlap = settings["chunking"]["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",           # cl100k_base encoding — works for all modern models
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        source = chunk.metadata.get("source_file", "unknown")
        chunk.metadata["chunk_id"] = f"{source}__chunk_{i}"

    avg_len = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"Created {len(chunks)} chunks  (avg {avg_len} chars each)")
    return chunks
