"""
BM25 keyword retriever using rank_bm25.

Ported and cleaned from Self-Corrective-Agentic-RAG/core/vector.py.
- Builds an in-memory BM25Okapi index over a document corpus
- Simple but effective tokenizer (lowercase, strip punctuation, whitespace split)
- update() supports incremental additions
- search() returns the top-k Documents ranked by BM25 score, zero-score docs filtered
"""

from __future__ import annotations

import re
import threading
from typing import List

from langchain_core.documents import Document
from loguru import logger

from config import RetrievalConfig


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Retriever:
    """Keyword (lexical) retriever over a fixed document corpus."""

    def __init__(self, documents: List[Document] | None = None):
        self._lock = threading.RLock()
        self.documents: List[Document] = []
        self._corpus: List[List[str]] = []
        self.bm25 = None
        if documents:
            self.update(documents)

    def update(self, new_docs: List[Document]) -> None:
        """Add documents and rebuild the BM25 index."""
        if not new_docs:
            return
        from rank_bm25 import BM25Okapi

        with self._lock:
            self.documents.extend(new_docs)
            self._corpus = [_tokenize(d.page_content) for d in self.documents]
            if self._corpus:
                self.bm25 = BM25Okapi(self._corpus)
                logger.info(f"BM25 index rebuilt with {len(self.documents)} documents")

    def search(self, query: str, k: int = RetrievalConfig.BM25_TOP_K) -> List[Document]:
        """Return top-k documents by BM25 score (zero-score docs filtered out)."""
        if not self.bm25 or not query.strip():
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            [(i, float(s)) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1], reverse=True,
        )[:k]
        results: List[Document] = []
        for idx, score in ranked:
            doc = self.documents[idx]
            meta = {**doc.metadata, "bm25_score": score, "retrieval_method": "bm25"}
            results.append(Document(page_content=doc.page_content, metadata=meta))
        return results

    def clear(self) -> None:
        with self._lock:
            self.documents = []
            self._corpus = []
            self.bm25 = None


# Module-level singleton
_bm25: BM25Retriever | None = None
_bm25_lock = threading.Lock()


def get_bm25_retriever() -> BM25Retriever:
    global _bm25
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                _bm25 = BM25Retriever()
    return _bm25
