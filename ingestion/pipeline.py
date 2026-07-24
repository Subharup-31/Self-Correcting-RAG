"""
Ingestion pipeline orchestrator.

Ties together:
  DocumentLoader → ParentChildChunker → VectorStore (children) + BM25Retriever

Provides:
  - ingest_file(path): ingest a single file
  - ingest_directory(path): ingest all supported files in a directory
  - ingest_documents(docs): ingest already-loaded LangChain Documents
  - reset(): wipe both stores

Returns a summary dict with counts and timings.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger

from ingestion.chunking import ParentChildChunker
from ingestion.document_loader import DocumentLoader
from ingestion.parent_store import save_parents, clear_parents
from retrieval.bm25_retriever import get_bm25_retriever
from retrieval.vector_store import get_vector_store


def ingest_documents(
    documents: List[Document],
    owner_id: str = "default_owner",
    session_id: str = "",
    persistent: bool = True
) -> dict:
    """Chunk and index a list of already-loaded Documents with owner and session metadata."""
    if not documents:
        return {"loaded": 0, "parents": 0, "children": 0, "seconds": 0.0}

    import uuid
    import datetime

    start = time.time()
    
    # Enrich metadata for each document
    for doc in documents:
        doc.metadata["owner_id"] = owner_id
        doc.metadata["session_id"] = session_id
        doc.metadata["persistent"] = persistent
        doc.metadata["uploaded_at"] = doc.metadata.get(
            "uploaded_at",
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        if "document_id" not in doc.metadata:
            doc.metadata["document_id"] = str(uuid.uuid4())
        # Track filename if path is available
        if "source" not in doc.metadata:
            doc.metadata["source"] = doc.metadata.get("file_name", "unknown")

    chunker = ParentChildChunker()
    children, parents = chunker.chunk(documents)

    # Save parents to SQLite store
    save_parents(parents)

    vs = get_vector_store()
    vs.add_documents(children)

    bm25 = get_bm25_retriever()
    bm25.update(children)  # BM25 over child chunks for precise keyword match

    elapsed = time.time() - start
    summary = {
        "loaded": len(documents),
        "parents": len(parents),
        "children": len(children),
        "vector_store_count": vs.count(),
        "bm25_count": len(bm25.documents),
        "seconds": round(elapsed, 2),
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


def ingest_file(
    path: str,
    owner_id: str = "default_owner",
    session_id: str = "",
    persistent: bool = True
) -> dict:
    """Load, chunk, and index a single file with context metadata."""
    logger.info(f"---INGEST FILE: {path}---")
    docs = DocumentLoader().load(path)
    if not docs:
        return {"loaded": 0, "parents": 0, "children": 0, "seconds": 0.0,
                "error": "no documents extracted"}
    return ingest_documents(docs, owner_id=owner_id, session_id=session_id, persistent=persistent)


def ingest_directory(
    path: str,
    owner_id: str = "default_owner",
    session_id: str = "",
    persistent: bool = True
) -> dict:
    """Load, chunk, and index all supported files in a directory with context metadata."""
    logger.info(f"---INGEST DIRECTORY: {path}---")
    docs = DocumentLoader().load_directory(path)
    if not docs:
        return {"loaded": 0, "parents": 0, "children": 0, "seconds": 0.0,
                "error": "no documents found"}
    return ingest_documents(docs, owner_id=owner_id, session_id=session_id, persistent=persistent)


def reset_stores() -> None:
    """Wipe both the vector store, parent store and BM25 index."""
    get_vector_store().clear()
    get_bm25_retriever().clear()
    clear_parents()
    logger.info("All stores reset.")



def get_ingestion_stats() -> dict:
    """Return current store sizes."""
    vs = get_vector_store()
    bm25 = get_bm25_retriever()
    return {
        "vector_store_chunks": vs.count(),
        "bm25_chunks": len(bm25.documents),
    }
