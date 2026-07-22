"""
Cross-encoder reranker.

Ported from Self-Healing-RAG/backend/reranker.py. Uses
`cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers.

A cross-encoder scores (query, document) PAIRS jointly — far more accurate than
bi-encoder similarity at the cost of speed. We therefore use it as a second
stage: the hybrid retriever produces candidates, the cross-encoder reranks them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from loguru import logger

from config import ModelConfig


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading cross-encoder: {ModelConfig.RERANKER_MODEL}")
    return CrossEncoder(ModelConfig.RERANKER_MODEL)


class CrossEncoderReranker:
    """Rerank documents against a query with a cross-encoder model."""

    def __init__(self, model_name: str = ModelConfig.RERANKER_MODEL, top_k: int = ModelConfig.RERANKER_TOP_K):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None  # lazy load

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cross_encoder()
        return self._model

    def rerank(
        self, query: str, documents: List[Document], top_k: int = None
    ) -> List[Document]:
        """Rerank documents by cross-encoder relevance score.

        Returns at most `top_k` documents, each annotated with `rerank_score`.
        """
        if not documents or not query.strip():
            return documents[: (top_k or self.top_k)]
        top_k = top_k or self.top_k

        texts = [d.page_content for d in documents]
        pairs = [(query, t) for t in texts]
        try:
            scores = self.model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Cross-encoder predict failed, returning original order: {exc}")
            return documents[:top_k]

        # Sort documents by score, descending.
        ranked: List[Tuple[float, Document]] = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )
        results: List[Document] = []
        for score, doc in ranked[:top_k]:
            meta = {**doc.metadata, "rerank_score": float(score)}
            results.append(Document(page_content=doc.page_content, metadata=meta))
        logger.info(
            f"Cross-encoder rerank: {len(documents)} → {len(results)} docs "
            f"(top score {ranked[0][0]:.3f})" if ranked else "rerank empty"
        )
        return results

    def rerank_with_scores(self, query: str, documents: List[Document], top_k: int = None):
        """Return (Document, score) tuples."""
        reranked = self.rerank(query, documents, top_k=top_k)
        return [(d, d.metadata.get("rerank_score", 0.0)) for d in reranked]


# Singleton
_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
