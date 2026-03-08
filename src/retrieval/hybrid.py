import hashlib
import re
from typing import Dict, List, Tuple

import yaml
from langchain_core.documents import Document

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score for a document d = Σ  1 / (k + rank(d, list_i))

    k=60 is the standard constant that dampens extreme rank differences.
    Documents are deduplicated by content hash so that identical text appearing
    on multiple pages (e.g. repeated title pages) is counted only once.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list in results_list:
        for rank, (doc, _original_score) in enumerate(result_list):
            # Normalize whitespace then hash — catches identical text that differs
            # only in line breaks or spacing (e.g. repeated PDF title pages).
            doc_id = hashlib.md5(
                re.sub(r"\s+", " ", doc.page_content).strip().encode()
            ).hexdigest()
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[did], scores[did]) for did in sorted_ids]


class HybridRetriever:
    """
    Combines vector-based semantic search and BM25 keyword search via RRF.

    Why both?
    - Vector search: captures semantic/conceptual similarity.
    - BM25 search:   captures exact keyword matches (e.g. function names, version numbers).
    Together they cover cases that either alone would miss.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
    ):
        settings = _load_settings()
        self._vector_store = vector_store
        self._bm25 = bm25_retriever
        self._vector_top_k = settings["retrieval"]["vector_top_k"]
        self._bm25_top_k   = settings["retrieval"]["bm25_top_k"]
        self._rrf_k        = settings["retrieval"]["rrf_k"]
        # Pass a generous pool to the reranker; it trims down to final_top_k.
        self._pool_size    = settings["retrieval"]["final_top_k"] * 4

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        vector_results = self._vector_store.similarity_search(query, k=self._vector_top_k)
        bm25_results   = self._bm25.search(query, k=self._bm25_top_k)

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self._rrf_k,
        )
        return fused[: self._pool_size]
