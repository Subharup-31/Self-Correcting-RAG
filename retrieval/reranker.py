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
        if ModelConfig.RERANKER_PROVIDER == "none":
            return None
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

        if ModelConfig.RERANKER_PROVIDER == "none":
            logger.info("Reranker provider is 'none', bypassing cross-encoder rerank (passthrough).")
            results = []
            for i, doc in enumerate(documents[:top_k]):
                score = 1.0 - (i / len(documents)) if len(documents) > 1 else 1.0
                meta = {**doc.metadata, "rerank_score": float(score)}
                results.append(Document(page_content=doc.page_content, metadata=meta))
            return results

        if ModelConfig.RERANKER_PROVIDER == "nvidia":
            import os
            import httpx
            from config import APIKeys
            
            logger.info(f"NVIDIA cloud reranker active: {self.model_name}")
            api_key = APIKeys.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")
            if not api_key:
                logger.error("NVIDIA_API_KEY is not set. Bypassing reranking.")
                return documents[:top_k]
                
            url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "query": {"text": query},
                "passages": [{"text": doc.page_content} for doc in documents]
            }
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                rankings = data.get("rankings", [])
                
                # Sort rankings by logit descending
                rankings_sorted = sorted(rankings, key=lambda x: x.get("logit", 0.0), reverse=True)
                
                results = []
                for item in rankings_sorted[:top_k]:
                    idx = item.get("index")
                    if idx is not None and 0 <= idx < len(documents):
                        doc = documents[idx]
                        logit = item.get("logit", 0.0)
                        meta = {**doc.metadata, "rerank_score": float(logit)}
                        results.append(Document(page_content=doc.page_content, metadata=meta))
                
                logger.info(f"NVIDIA Rerank complete: {len(documents)} -> {len(results)} docs")
                return results
            except Exception as exc:
                logger.warning(f"NVIDIA rerank failed, keeping original order: {exc}")
                return documents[:top_k]

        if ModelConfig.RERANKER_PROVIDER == "openrouter":
            import os
            import httpx
            from config import APIKeys
            
            logger.info(f"OpenRouter cloud reranker active: {self.model_name}")
            api_key = APIKeys.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.error("OPENROUTER_API_KEY is not set. Bypassing reranking.")
                return documents[:top_k]
                
            url = "https://openrouter.ai/api/v1/rerank"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": [doc.page_content for doc in documents],
                "top_n": top_k
            }
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                results_list = data.get("results", [])
                
                # Sort rankings by relevance_score descending
                results_sorted = sorted(results_list, key=lambda x: x.get("relevance_score", 0.0), reverse=True)
                
                results = []
                for item in results_sorted[:top_k]:
                    idx = item.get("index")
                    if idx is not None and 0 <= idx < len(documents):
                        doc = documents[idx]
                        score = item.get("relevance_score", 0.0)
                        meta = {**doc.metadata, "rerank_score": float(score)}
                        results.append(Document(page_content=doc.page_content, metadata=meta))
                
                logger.info(f"OpenRouter Rerank complete: {len(documents)} -> {len(results)} docs")
                return results
            except Exception as exc:
                logger.warning(f"OpenRouter rerank failed, keeping original order: {exc}")
                return documents[:top_k]

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
