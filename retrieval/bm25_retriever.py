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

from config import BASE_DIR, RetrievalConfig


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
        self._corpus_version: int = 0
        self._isolated_cache: dict = {}  # (owner_id, session_id) -> (version, filtered_docs, BM25Okapi)
        if documents:
            self.update(documents)
        else:
            self.load_from_disk()

    def _invalidate_cache(self) -> None:
        """Bump corpus version and purge cached tenant indexes."""
        self._corpus_version += 1
        self._isolated_cache.clear()

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
                self.save_to_disk()
            self._invalidate_cache()

    def search(
        self,
        query: str,
        k: int = RetrievalConfig.BM25_TOP_K,
        owner_id: str = "default_owner",
        session_id: str = ""
    ) -> List[Document]:
        """Return top-k documents by BM25 score, scoped to owner and/or session documents."""
        if not self.documents or not query.strip():
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        # 1. Filter documents based on owner_id and session_id for isolation
        with self._lock:
            filtered_docs = []
            for doc in self.documents:
                meta = doc.metadata
                is_session = (session_id and meta.get("session_id") == session_id)
                is_owner_persistent = (owner_id and meta.get("owner_id") == owner_id and meta.get("persistent") is True)
                if is_session or is_owner_persistent:
                    filtered_docs.append(doc)

            if not filtered_docs:
                return []

            # 2. Reuse cached BM25Okapi index for (owner_id, session_id) if corpus version matches
            cache_key = (owner_id or "default_owner", session_id or "")
            cached = self._isolated_cache.get(cache_key)

            if cached and cached[0] == self._corpus_version and len(cached[1]) == len(filtered_docs):
                local_bm25 = cached[2]
            else:
                from rank_bm25 import BM25Okapi
                local_corpus = [_tokenize(d.page_content) for d in filtered_docs]
                local_bm25 = BM25Okapi(local_corpus)
                self._isolated_cache[cache_key] = (self._corpus_version, filtered_docs, local_bm25)

            scores = local_bm25.get_scores(tokens)

        ranked = sorted(
            [(i, float(s)) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1], reverse=True,
        )[:k]

        results: List[Document] = []
        for idx, score in ranked:
            doc = filtered_docs[idx]
            meta = {**doc.metadata, "bm25_score": score, "retrieval_method": "bm25"}
            results.append(Document(page_content=doc.page_content, metadata=meta))
        return results

    def save_to_disk(self, filepath: str = None) -> None:
        """Serialize documents to a pickle file on disk."""
        import pickle
        path = filepath or str(BASE_DIR / "bm25_store.pkl")
        with self._lock:
            try:
                with open(path, "wb") as f:
                    pickle.dump(self.documents, f)
                logger.info(f"Saved {len(self.documents)} documents to BM25 disk cache: {path}")
            except Exception as exc:
                logger.error(f"Failed to save BM25 retriever to disk: {exc}")

    def load_from_disk(self, filepath: str = None) -> bool:
        """Load documents from pickle and rebuild index."""
        import pickle
        import os
        path = filepath or str(BASE_DIR / "bm25_store.pkl")
        if not os.path.exists(path):
            return False
        with self._lock:
            try:
                with open(path, "rb") as f:
                    docs = pickle.load(f)
                self.documents = docs
                self._corpus = [_tokenize(d.page_content) for d in self.documents]
                if self._corpus:
                    from rank_bm25 import BM25Okapi
                    self.bm25 = BM25Okapi(self._corpus)
                    logger.info(f"Loaded {len(self.documents)} documents from BM25 disk cache and rebuilt index.")
                    self._invalidate_cache()
                    return True
            except Exception as exc:
                logger.error(f"Failed to load BM25 retriever from disk: {exc}")
        return False

    def clear(self) -> None:
        import os
        with self._lock:
            self.documents = []
            self._corpus = []
            self.bm25 = None
            self._invalidate_cache()
            path = str(BASE_DIR / "bm25_store.pkl")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Deleted BM25 disk cache file: {path}")
                except Exception as exc:
                    logger.warning(f"Could not delete BM25 disk cache: {exc}")

    def delete_session(self, session_id: str) -> None:
        """Purge all documents belonging to a session from BM25 index and save to disk."""
        if not session_id:
            return
        from rank_bm25 import BM25Okapi
        with self._lock:
            initial_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc.metadata.get("session_id") != session_id]
            self._corpus = [_tokenize(d.page_content) for d in self.documents]
            if self._corpus:
                self.bm25 = BM25Okapi(self._corpus)
            else:
                self.bm25 = None
            self._invalidate_cache()
            self.save_to_disk()
            logger.info(f"Purged session {session_id} from BM25 index (removed {initial_count - len(self.documents)} child chunks).")



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
