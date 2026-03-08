from typing import List, Tuple

from langchain_core.documents import Document

from src.generation.prompt_manager import PromptManager


class CitationGuard:
    """
    Gate between retrieval and generation.

    After cross-encoder reranking, the best chunk's score tells us how
    confident the retriever is. If the score is below the threshold, the
    retrieved evidence is too weak — the system declines to answer rather
    than risk producing a hallucinated or unsupported response.

    Threshold tuning guide (ms-marco-MiniLM-L-6-v2 scores):
      > 0    — clear topical match
      -2 to 0 — weak match, often borderline
      < -5   — off-topic / no real match  ← default decline boundary
    """

    def __init__(self, decline_threshold: float = -5.0):
        self._threshold   = decline_threshold
        self._prompt_mgr  = PromptManager()

    def check(
        self,
        reranked_chunks: List[Tuple[Document, float]],
    ) -> Tuple[bool, str]:
        """
        Returns (should_proceed, message).
        - should_proceed=True  → proceed to generation; message is empty.
        - should_proceed=False → decline; message contains the user-facing explanation.
        """
        if not reranked_chunks:
            return False, self._prompt_mgr.get_decline_message()

        best_score = reranked_chunks[0][1]
        if best_score < self._threshold:
            return False, self._prompt_mgr.get_decline_message()

        return True, ""
