from typing import List, Tuple

import yaml
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


class CrossEncoderReranker:
    """
    Re-scores (query, chunk) pairs with a cross-encoder model.

    Unlike bi-encoders (used in vector search), a cross-encoder sees the
    query and chunk *together*, enabling much finer relevance scoring.
    This is the single highest-ROI improvement over a basic vector RAG.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Trained on MS MARCO passage ranking
      - Output: raw logit (higher = more relevant)
      - Typical range: roughly -10 to +10
    """

    def __init__(self):
        settings = _load_settings()
        self._model_name      = settings["reranker"]["model"]
        self._top_n           = settings["reranker"]["top_n"]
        self._decline_threshold = settings["citation"]["decline_threshold"]

        print(f"Loading reranker: {self._model_name}")
        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Score every (query, chunk) pair and return the top_n sorted descending.
        The float in each tuple is now the cross-encoder logit, not the RRF score.
        """
        if not candidates:
            return []

        pairs  = [(query, doc.page_content) for doc, _ in candidates]
        scores = self._model.predict(pairs)

        reranked = sorted(
            zip([doc for doc, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked[: self._top_n]

    def should_decline(self, reranked: List[Tuple[Document, float]]) -> bool:
        """
        Returns True when the best chunk score is below the decline threshold,
        meaning the retrieved evidence is too weak to support a trustworthy answer.
        """
        if not reranked:
            return True
        return reranked[0][1] < self._decline_threshold
