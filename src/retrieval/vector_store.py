import warnings
from typing import List, Optional, Tuple

import yaml
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


class VectorStore:
    """
    Thin wrapper around ChromaDB for storing and querying document embeddings.
    Data is persisted to data/chroma_db/<collection_name>/.
    """

    def __init__(
        self,
        collection_name: str = "docs",
        embedding_function: Optional[HuggingFaceEmbeddings] = None,
    ):
        self.collection_name   = collection_name
        self.persist_directory = f"data/chroma_db/{collection_name}"

        if embedding_function is None:
            settings = _load_settings()
            embedding_function = HuggingFaceEmbeddings(
                model_name=settings["embedding"]["model"],
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        self._embedding_fn = embedding_function
        self._db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document]) -> None:
        self._db.add_documents(documents)

    def reset(self) -> None:
        """Delete and recreate the collection using ChromaDB's own API.

        Why not shutil.rmtree?
        ChromaDB >=0.5 uses a Rust-based SQLite connection pool. Deleting the
        file from disk while the Rust client still holds an open connection
        leaves the new Chroma instance unable to write ('readonly database').
        delete_collection() tells the Rust runtime to close the connection
        cleanly before we reinitialise.
        """
        try:
            self._db.delete_collection()
            print(f"  Cleared collection '{self.collection_name}'")
        except Exception:
            # Collection didn't exist yet — nothing to clear, proceed as normal.
            print(f"  Collection '{self.collection_name}' not found — starting fresh")

        self._db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embedding_fn,
            persist_directory=self.persist_directory,
        )

    def similarity_search(
        self, query: str, k: int = 20
    ) -> List[Tuple[Document, float]]:
        """Return (Document, cosine_similarity_score) pairs, sorted descending.
        Negative scores (unrelated queries) are clipped to 0.0 — they carry
        no useful signal and cause a noisy ChromaDB UserWarning otherwise.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self._db.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, max(0.0, score)) for doc, score in results]
