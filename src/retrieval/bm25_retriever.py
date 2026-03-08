import json
import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based retriever using BM25Okapi.
    Loads pre-saved chunks from data/processed/<collection_name>_chunks.json
    so it shares the exact same corpus as the vector store.
    """

    def __init__(self, collection_name: str = "docs"):
        self.collection_name = collection_name
        self.chunks: List[Document] = []
        self._index: BM25Okapi
        self._build_index(collection_name)

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lowercase + strip punctuation + whitespace split."""
        return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()

    def _build_index(self, collection_name: str) -> None:
        path = f"data/processed/{collection_name}_chunks.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in data
        ]

        tokenised_corpus = [self._tokenise(c.page_content) for c in self.chunks]
        self._index = BM25Okapi(tokenised_corpus)
        print(f"BM25 index ready — {len(self.chunks)} chunks")

    # ── public ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Return (Document, bm25_score) pairs, top-k sorted descending."""
        tokens = self._tokenise(query)
        scores = self._index.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
