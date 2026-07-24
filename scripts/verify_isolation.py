"""
Verification script for multi-user document isolation and session cleanup.

Verifies:
1. Watertight user document isolation (User A cannot see User B's documents).
2. Session-based temporary document scoping (Session documents only visible to that session).
3. Automatic cleanup of session documents (temporary documents purged, persistent ones remain).
4. Correct function of parent-child retrieval, BM25, and hybrid retrievers with metadata filters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import BASE_DIR
from ingestion.pipeline import ingest_documents, reset_stores
from ingestion.lifecycle_manager import DocumentLifecycleManager
from retrieval.vector_store import get_vector_store
from retrieval.bm25_retriever import get_bm25_retriever
from retrieval.hybrid_retriever import HybridRetriever
from langchain_core.documents import Document


def main():
    logger.info("=== STARTING RETRIEVAL ISOLATION VERIFICATION ===")

    # 1. Reset all stores to starting state
    logger.info("Resetting stores...")
    reset_stores()

    # 2. Define synthetic test documents
    doc_a = Document(
        page_content="User A specific secret project codename is: PROJECT_NEO_ALPHA. This is user a persistent document.",
        metadata={"file_name": "user_a_secret.txt"}
    )
    doc_b = Document(
        page_content="User B specific secret project codename is: PROJECT_ZEUS_GAMMA. This is user b persistent document.",
        metadata={"file_name": "user_b_secret.txt"}
    )
    doc_temp = Document(
        page_content="Session temporary notes containing passcode: SHIELD_99812_THETA. This is a temporary session note.",
        metadata={"file_name": "session_temp_notes.txt"}
    )

    # 3. Ingest documents with metadata
    logger.info("Ingesting User A persistent document...")
    ingest_documents([doc_a], owner_id="user_a", session_id="", persistent=True)

    logger.info("Ingesting User B persistent document...")
    ingest_documents([doc_b], owner_id="user_b", session_id="", persistent=True)

    logger.info("Ingesting Session temporary document (User A, Session 'sess_123')...")
    ingest_documents([doc_temp], owner_id="user_a", session_id="sess_123", persistent=False)

    # 4. Instantiate retrievers
    vs = get_vector_store()
    bm25 = get_bm25_retriever()
    hybrid = HybridRetriever(vector_store=vs, bm25=bm25)

    # 5. Run isolation assertions

    # Case A: User A querying alone (without active session)
    logger.info("--- TEST 1: User A alone ---")
    results = hybrid.retrieve("PROJECT_NEO_ALPHA", owner_id="user_a", session_id="")
    assert len(results) > 0, "User A should retrieve their own persistent documents."
    assert any("PROJECT_NEO_ALPHA" in r.page_content for r in results), "Neo Alpha should be retrieved"
    
    # Confirm isolation: User A cannot retrieve User B's document
    results_cross = hybrid.retrieve("PROJECT_ZEUS_GAMMA", owner_id="user_a", session_id="")
    assert not any("PROJECT_ZEUS_GAMMA" in r.page_content for r in results_cross), \
        "SECURITY BREACH: User A retrieved User B's document!"

    # Case B: User B querying alone
    logger.info("--- TEST 2: User B alone ---")
    results_b = hybrid.retrieve("PROJECT_ZEUS_GAMMA", owner_id="user_b", session_id="")
    assert len(results_b) > 0, "User B should retrieve their own persistent documents."
    assert any("PROJECT_ZEUS_GAMMA" in r.page_content for r in results_b), "Zeus Gamma should be retrieved"

    # Confirm isolation: User B cannot retrieve User A's document or session document
    results_b_cross = hybrid.retrieve("PROJECT_NEO_ALPHA", owner_id="user_b", session_id="")
    assert not any("PROJECT_NEO_ALPHA" in r.page_content for r in results_b_cross), \
        "SECURITY BREACH: User B retrieved User A's document!"
    results_b_temp = hybrid.retrieve("SHIELD_99812_THETA", owner_id="user_b", session_id="")
    assert not any("SHIELD_99812_THETA" in r.page_content for r in results_b_temp), \
        "SECURITY BREACH: User B retrieved Session temporary document!"

    # Case C: User A querying with Session 'sess_123'
    logger.info("--- TEST 3: User A with Session 'sess_123' ---")
    results_sess = hybrid.retrieve("SHIELD_99812_THETA", owner_id="user_a", session_id="sess_123")
    assert len(results_sess) > 0, "User A in sess_123 should retrieve session documents."
    assert any("SHIELD_99812_THETA" in r.page_content for r in results_sess), "Session passcode should be retrieved"

    # Case D: User A querying with Session 'sess_123' should also retrieve their persistent documents (UNION mode)
    results_union = hybrid.retrieve("PROJECT_NEO_ALPHA", owner_id="user_a", session_id="sess_123")
    assert len(results_union) > 0, "User A in sess_123 should retrieve persistent documents."

    # Case E: User A querying with a different session ID should NOT get sess_123 documents
    logger.info("--- TEST 4: User A with different Session ---")
    results_diff_sess = hybrid.retrieve("SHIELD_99812_THETA", owner_id="user_a", session_id="sess_999")
    assert not any("SHIELD_99812_THETA" in r.page_content for r in results_diff_sess), \
        "SECURITY BREACH: User retrieved documents from a different session!"

    # 6. Session Cleanup Verification
    logger.info("Purging Session 'sess_123'...")
    cleanup_result = DocumentLifecycleManager.cleanup_session("sess_123")
    assert cleanup_result["status"] == "success"

    # Verify session documents are deleted
    logger.info("Checking session deletion...")
    results_post_cleanup = hybrid.retrieve("SHIELD_99812_THETA", owner_id="user_a", session_id="sess_123")
    assert not any("SHIELD_99812_THETA" in r.page_content for r in results_post_cleanup), \
        "CLEANUP FAILURE: Session document still exists after cleanup!"

    # Verify persistent documents survive
    logger.info("Checking persistent document survival...")
    results_survive = hybrid.retrieve("PROJECT_NEO_ALPHA", owner_id="user_a", session_id="sess_123")
    assert len(results_survive) > 0, "Persistent document should survive session cleanup!"
    assert any("PROJECT_NEO_ALPHA" in r.page_content for r in results_survive), "Neo Alpha should survive"

    logger.info("=== ALL ISOLATION & CLEANUP TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
