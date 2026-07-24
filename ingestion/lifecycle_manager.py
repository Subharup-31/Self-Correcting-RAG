"""
Document lifecycle and session manager.

Handles multi-user document isolation and session-temporary document cleanup.
Wipes session vectors, parent store records, lexical index items, and temporary disk files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from loguru import logger

from config import BASE_DIR
from ingestion.parent_store import delete_session_parents
from retrieval.bm25_retriever import get_bm25_retriever
from retrieval.vector_store import get_vector_store


class DocumentLifecycleManager:
    """Manages document lifecycles and provides automatic cleanup of temporary session documents."""

    @staticmethod
    def cleanup_session(session_id: str) -> dict:
        """Purge all vector database, parent SQLite database, and BM25 index points matching session_id."""
        if not session_id or not session_id.strip():
            return {"status": "ignored", "reason": "empty session_id"}

        logger.info(f"=== INITIATING LIFECYCLE CLEANUP FOR SESSION: {session_id} ===")
        purged = {}

        # 1. Clean Vector DB (Qdrant or Chroma)
        try:
            vs = get_vector_store()
            from langchain_qdrant import QdrantVectorStore
            from langchain_chroma import Chroma

            if isinstance(vs.store, QdrantVectorStore):
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                client = vs.store.client
                collection_name = vs.collection_name
                # Delete points matching metadata.session_id == session_id
                client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.session_id",
                                match=MatchValue(value=session_id)
                            )
                        ]
                    )
                )
                logger.info(f"Qdrant session {session_id} points deleted.")
                purged["vector_store"] = "Qdrant points deleted"
            elif isinstance(vs.store, Chroma):
                vs.store.delete(where={"session_id": session_id})
                logger.info(f"Chroma session {session_id} points deleted.")
                purged["vector_store"] = "Chroma points deleted"
            else:
                if hasattr(vs.store, "delete"):
                    vs.store.delete(where={"session_id": session_id})
                    purged["vector_store"] = "LangChain delete invoked"
        except Exception as exc:
            logger.error(f"Failed to delete session {session_id} from vector store: {exc}")
            purged["vector_store_error"] = str(exc)

        # 2. Clean SQLite Parent Store
        try:
            delete_session_parents(session_id)
            purged["parent_store"] = "SQLite parent chunks deleted"
        except Exception as exc:
            logger.error(f"Failed to delete session {session_id} parents from SQLite: {exc}")
            purged["parent_store_error"] = str(exc)

        # 3. Clean BM25 Index
        try:
            get_bm25_retriever().delete_session(session_id)
            purged["bm25"] = "BM25 index updated"
        except Exception as exc:
            logger.error(f"Failed to delete session {session_id} from BM25 index: {exc}")
            purged["bm25_error"] = str(exc)

        # 4. Cleanup temporary folder structures
        try:
            session_temp_dir = BASE_DIR / "documents" / f"session_{session_id}"
            if session_temp_dir.exists() and session_temp_dir.is_dir():
                shutil.rmtree(session_temp_dir)
                logger.info(f"Removed session directory: {session_temp_dir}")
                purged["temp_directory"] = "Deleted"
        except Exception as exc:
            logger.error(f"Failed to delete session {session_id} temp directories: {exc}")
            purged["temp_directory_error"] = str(exc)

        logger.info(f"=== CLEANUP FOR SESSION {session_id} COMPLETE ===")
        return {"status": "success", "session_id": session_id, "purged": purged}
