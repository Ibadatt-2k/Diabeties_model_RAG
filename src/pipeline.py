import hashlib
import re
import yaml

from src.generation.citation_guard import CitationGuard
from src.generation.generator import RAGGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import VectorStore


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """
    Single entry point for querying the RAG system.

    phase=1 — Phase 1 (fundamentals):
        vector similarity search  →  generate answer

    phase=2 — Phase 2 (production):
        hybrid retrieval (vector + BM25 via RRF)
        → cross-encoder reranking
        → citation guard (decline if evidence is too weak)
        → generate answer
    """

    def __init__(self, collection_name: str = "docs", phase: int = 2):
        self._phase           = phase
        self._settings        = _load_settings()
        self._collection_name = collection_name

        print(f"Initialising RAG Pipeline  (phase={phase}, collection='{collection_name}')")

        # ── always needed ─────────────────────────────────────────────────────
        self._vector_store = VectorStore(collection_name=collection_name)
        self._generator    = RAGGenerator()

        # ── phase 2 only ─────────────────────────────────────────────────────
        if phase >= 2:
            self._bm25     = BM25Retriever(collection_name=collection_name)
            self._hybrid   = HybridRetriever(self._vector_store, self._bm25)
            self._reranker = CrossEncoderReranker()
            self._guard    = CitationGuard(
                decline_threshold=self._settings["citation"]["decline_threshold"]
            )

        print("Pipeline ready.\n")

    # ── public ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """Run a question through the pipeline and return a structured result."""
        if self._phase == 1:
            return self._phase1(question)
        return self._phase2(question)

    # ── private ───────────────────────────────────────────────────────────────

    def _phase1(self, question: str) -> dict:
        k   = self._settings["retrieval"]["final_top_k"]
        # Fetch a wider pool then deduplicate by content hash.
        # chunk_id dedup alone misses identical content on different pages
        # (e.g. a title repeated on page 1, 2, and 74 of the same PDF).
        raw = self._vector_store.similarity_search(question, k=k * 4)

        seen_hashes: set = set()
        chunks = []
        for doc, score in raw:
            content_hash = hashlib.md5(
                re.sub(r"\s+", " ", doc.page_content).strip().encode()
            ).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                chunks.append((doc, score))
            if len(chunks) >= k:
                break

        if not chunks:
            return {"answer": "No relevant documents found.", "sources": [], "phase": 1}

        result             = self._generator.generate(question, chunks)
        result["phase"]    = 1
        result["declined"] = False
        return result

    def _phase2(self, question: str) -> dict:
        # 1. Hybrid retrieval
        candidates = self._hybrid.retrieve(question)

        # 2. Cross-encoder reranking
        reranked = self._reranker.rerank(question, candidates)

        # 3. Citation guard
        should_proceed, decline_msg = self._guard.check(reranked)
        if not should_proceed:
            return {
                "answer":   decline_msg,
                "sources":  [],
                "declined": True,
                "phase":    2,
            }

        # 4. Generate
        result             = self._generator.generate(question, reranked)
        result["declined"] = False
        result["phase"]    = 2
        return result
