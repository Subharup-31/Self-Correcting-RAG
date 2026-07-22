"""
Hybrid retriever combining BM25 + vector search via Reciprocal Rank Fusion.

RRF formula (standard k=60):
    score(d) = sum_i  1 / (k + rank_i(d))

Document identity across the two ranking lists is established by an MD5 hash of
the page_content (ported from Self-Corrective-Agentic-RAG) so that the same
chunk appearing in both lists has its rank-credit accumulated.
"""

from __future__ import annotations

import hashlib
from typing import List

from langchain_core.documents import Document
from loguru import logger

from config import RetrievalConfig
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_store import VectorStore


def _doc_key(doc: Document) -> str:
    """Stable identity key for a document (MD5 of content)."""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], k: int = RetrievalConfig.RRF_K
) -> List[Document]:
    """Fuse multiple ranked document lists into one via RRF."""
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = _doc_key(doc)
            if key not in fused_scores:
                fused_scores[key] = 0.0
                doc_map[key] = doc
            fused_scores[key] += 1.0 / (k + rank)

    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    fused: List[Document] = []
    for key in sorted_keys:
        doc = doc_map[key]
        meta = {**doc.metadata, "fusion_score": fused_scores[key],
                "retrieval_method": "hybrid_rrf"}
        fused.append(Document(page_content=doc.page_content, metadata=meta))
    return fused


class HybridRetriever:
    """BM25 + ChromaDB vector search fused with RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        k: int = RetrievalConfig.RRF_K,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.k = k

    def retrieve(self, query: str, top_k: int = RetrievalConfig.HYBRID_FINAL_K) -> List[Document]:
        """Run both retrievers and fuse their results."""
        if not query.strip():
            return []

        # Vector (semantic) retrieval.
        try:
            vector_results = self.vector_store.search(query, k=RetrievalConfig.VECTOR_TOP_K)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Vector search failed: {exc}")
            vector_results = []
        for d in vector_results:
            d.metadata.setdefault("retrieval_method", "vector")

        # BM25 (lexical) retrieval.
        bm25_results = self.bm25.search(query, k=RetrievalConfig.BM25_TOP_K)

        # Fuse.
        ranked_lists = []
        if vector_results:
            ranked_lists.append(vector_results)
        if bm25_results:
            ranked_lists.append(bm25_results)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, k=self.k)
        logger.info(
            f"Hybrid retrieve '{query[:40]}...': "
            f"{len(vector_results)} vector, {len(bm25_results)} bm25 → "
            f"{len(fused)} fused (returning {min(top_k, len(fused))})"
        )
        return fused[:top_k]
