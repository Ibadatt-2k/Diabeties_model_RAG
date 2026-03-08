import json
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def get_embedding_model() -> HuggingFaceEmbeddings:
    settings = _load_settings()
    model_name = settings["embedding"]["model"]
    print(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_and_store(
    chunks: List[Document],
    collection_name: str = "docs",
    reset: bool = False,
) -> None:
    """
    Embed all chunks and store them in ChromaDB.
    Also serialises the raw chunks to JSON so BM25 can load them without re-embedding.

    Pass reset=True to wipe the existing collection before storing.
    Using a single VectorStore instance for both reset and add_documents avoids
    the ChromaDB 'readonly database' error caused by two concurrent Chroma writers.
    """
    from src.retrieval.vector_store import VectorStore   # avoid circular import at module load

    embedding_model = get_embedding_model()
    store = VectorStore(collection_name=collection_name, embedding_function=embedding_model)

    if reset:
        print("  Resetting collection (clearing existing data)...")
        store.reset()

    print(f"Storing {len(chunks)} chunks in ChromaDB collection '{collection_name}'...")
    store.add_documents(chunks)

    # Persist chunks as JSON for BM25 (no embeddings needed)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    json_path = processed_dir / f"{collection_name}_chunks.json"

    serialised = [
        {"content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)

    print(f"Chunks also saved to {json_path}  (used by BM25 retriever)")
