"""
Cross-encoder rerank node.

Runs the cross-encoder reranker over retrieved documents to refine ordering
before generation. This is a quality-boosting step: the hybrid retriever gives
candidates; the cross-encoder scores (query, doc) pairs jointly for precision.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from loguru import logger

from graph.state import GraphState
from retrieval.reranker import CrossEncoderReranker, get_reranker

RERANK = "rerank"


def rerank(state: GraphState, reranker: CrossEncoderReranker = None,
           top_k: int = 5) -> GraphState:
    """Rerank retrieved documents with the cross-encoder."""
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    if not documents:
        return {}

    reranker = reranker or get_reranker()
    logger.info(f"---RERANK--- {len(documents)} docs")
    try:
        reranked = reranker.rerank(question, documents, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Rerank failed, keeping original order: {exc}")
        reranked = documents[:top_k]

    techniques = list(state.get("techniques_used", []))
    if "Cross-Encoder Reranking" not in techniques:
        techniques.append("Cross-Encoder Reranking")

    logger.info(f"---RERANKED → {len(reranked)} docs---")
    return {"documents": reranked, "techniques_used": techniques}
