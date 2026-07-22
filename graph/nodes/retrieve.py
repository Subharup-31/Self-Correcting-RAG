"""
Retrieve node.

Runs hybrid retrieval (BM25 + vector + RRF), optionally with HyDE.
Query decomposition results (if any) are used to run multiple sub-retrievals
whose results are merged and deduplicated.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from loguru import logger

from graph.state import GraphState
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.hyde import HyDERetriever

RETRIEVE = "retrieve"


def retrieve(state: GraphState, retriever: HybridRetriever,
             hyde: HyDERetriever | None = None,
             use_hyde: bool = True) -> GraphState:
    """Retrieve documents for the question (and any sub-questions).

    Multi-angle retrieval: for each sub-question (if decomposed) we retrieve,
    then merge + dedupe by content. HyDE is used on the primary question when
    enabled.
    """
    question = state["question"]
    sub_questions: List[str] = state.get("sub_questions", []) or []
    techniques = list(state.get("techniques_used", []))

    logger.info(f"---RETRIEVE--- question='{question[:50]}' subs={len(sub_questions)}")

    all_docs: List[Document] = []
    seen_keys = set()

    def _collect(docs: List[Document]) -> None:
        for d in docs:
            key = d.page_content[:200]  # dedupe key
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(d)

    # Primary retrieval — HyDE first (more precise), plain hybrid as backup.
    if use_hyde and hyde is not None:
        try:
            docs = hyde.retrieve(question, top_k=6)
            _collect(docs)
            if "HyDE" not in techniques:
                techniques.append("HyDE")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"HyDE failed, using plain hybrid: {exc}")
            _collect(retriever.retrieve(question, top_k=6))
    else:
        _collect(retriever.retrieve(question, top_k=6))

    # Sub-question retrieval (from query decomposition).
    for sq in sub_questions:
        if sq.strip() and sq != question:
            try:
                _collect(retriever.retrieve(sq, top_k=4))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Sub-question retrieve failed for '{sq}': {exc}")

    if "Hybrid Retrieval (BM25+Vector+RRF)" not in techniques:
        techniques.append("Hybrid Retrieval (BM25+Vector+RRF)")

    logger.info(f"---RETRIEVED {len(all_docs)} unique documents---")
    return {"documents": all_docs, "techniques_used": techniques}
