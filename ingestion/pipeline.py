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
from retrieval.bm25_retriever import get_bm25_retriever
from retrieval.vector_store import get_vector_store


def ingest_documents(documents: List[Document]) -> dict:
    """Chunk and index a list of already-loaded Documents."""
    if not documents:
        return {"loaded": 0, "parents": 0, "children": 0, "seconds": 0.0}

    start = time.time()
    chunker = ParentChildChunker()
    children, parents = chunker.chunk(documents)

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


def ingest_file(path: str) -> dict:
    """Load, chunk, and index a single file."""
    logger.info(f"---INGEST FILE: {path}---")
    docs = DocumentLoader().load(path)
    if not docs:
        return {"loaded": 0, "parents": 0, "children": 0, "seconds": 0.0,
                "error": "no documents extracted"}
    return ingest_documents(docs)


def ingest_directory(path: str) -> dict:
    """Load, chunk, and index all supported files in a directory."""
    logger.info(f"---INGEST DIRECTORY: {path}---")
    docs = DocumentLoader().load_directory(path)
    if not docs:
        return {"loaded": 0, "parents": 0, "children": 0, "seconds": 0.0,
                "error": "no documents found"}
    return ingest_documents(docs)


def reset_stores() -> None:
    """Wipe both the vector store and BM25 index."""
    get_vector_store().clear()
    get_bm25_retriever().clear()
    logger.info("All stores reset.")


def get_ingestion_stats() -> dict:
    """Return current store sizes."""
    vs = get_vector_store()
    bm25 = get_bm25_retriever()
    return {
        "vector_store_chunks": vs.count(),
        "bm25_chunks": len(bm25.documents),
    }
