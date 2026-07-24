"""
SQLite-backed store for parent document chunks.

Provides persistent mapping of `parent_id` -> parent `Document` (text + metadata),
ensuring we can resolve child chunks back to their full parent context at query time,
while enforcing multi-user and session isolation filters.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from config import BASE_DIR

DB_PATH = BASE_DIR / "parent_chunks.db"
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    # Enable WAL (Write-Ahead Log) for concurrent read/write support
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Initialize the SQLite parent chunk database and tables with migration."""
    with _lock:
        with _get_conn() as conn:
            # Create table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    parent_id TEXT PRIMARY KEY,
                    page_content TEXT,
                    metadata TEXT,
                    owner_id TEXT,
                    session_id TEXT,
                    persistent INTEGER DEFAULT 1
                )
            """)
            # Migrations: check if columns exist
            cursor = conn.execute("PRAGMA table_info(parent_chunks)")
            columns = [row[1] for row in cursor.fetchall()]
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE parent_chunks ADD COLUMN owner_id TEXT")
            if "session_id" not in columns:
                conn.execute("ALTER TABLE parent_chunks ADD COLUMN session_id TEXT")
            if "persistent" not in columns:
                conn.execute("ALTER TABLE parent_chunks ADD COLUMN persistent INTEGER DEFAULT 1")
            conn.commit()


# Automatically initialize the database on import
try:
    init_db()
except Exception as exc:
    logger.error(f"Failed to initialize parent store DB: {exc}")


def save_parents(parents: List[Document]) -> None:
    """Save parent documents to the SQLite database with user/session metadata."""
    if not parents:
        return

    valid_parents = []
    for p in parents:
        pid = p.metadata.get("parent_id")
        if not pid:
            continue
        try:
            meta_str = json.dumps(p.metadata)
            owner_id = p.metadata.get("owner_id", "default_owner")
            session_id = p.metadata.get("session_id", "")
            persistent = 1 if p.metadata.get("persistent", True) else 0
            valid_parents.append((pid, p.page_content, meta_str, owner_id, session_id, persistent))
        except Exception as exc:
            logger.warning(f"Failed to serialize metadata for parent {pid}: {exc}")

    if not valid_parents:
        return

    with _lock:
        try:
            with _get_conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO parent_chunks (parent_id, page_content, metadata, owner_id, session_id, persistent) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    valid_parents
                )
                conn.commit()
            logger.info(f"Saved {len(valid_parents)} parent document(s) to SQLite store.")
        except Exception as exc:
            logger.error(f"Failed to save parents to database: {exc}")


def get_parents(parent_ids: List[str], owner_id: str = "default_owner", session_id: str = "") -> List[Document]:
    """Retrieve parent documents for a list of parent_ids, enforcing owner_id/session_id isolation."""
    if not parent_ids:
        return []

    unique_ids = list(set(pid for pid in parent_ids if pid))
    if not unique_ids:
        return []

    results: List[Document] = []
    with _lock:
        try:
            with _get_conn() as conn:
                # Query in batches to stay well under the SQLite parameter limit (999)
                for i in range(0, len(unique_ids), 400):
                    batch = unique_ids[i: i + 400]
                    placeholders = ",".join("?" for _ in batch)
                    query = f"""
                        SELECT parent_id, page_content, metadata FROM parent_chunks
                        WHERE parent_id IN ({placeholders})
                        AND (
                            (session_id = ? AND session_id != '')
                            OR (owner_id = ? AND persistent = 1)
                        )
                    """
                    params = list(batch) + [session_id or "__NO_SESSION__", owner_id]
                    cursor = conn.execute(query, params)
                    for _, content, meta_str in cursor:
                        try:
                            metadata = json.loads(meta_str)
                        except Exception:
                            metadata = {}
                        results.append(Document(page_content=content, metadata=metadata))
        except Exception as exc:
            logger.error(f"Failed to query parent store: {exc}")

    return results


def get_parent_by_id(parent_id: str, owner_id: str = "default_owner", session_id: str = "") -> Optional[Document]:
    """Retrieve a single parent document by its parent_id, enforcing owner_id/session_id isolation."""
    if not parent_id:
        return None

    with _lock:
        try:
            with _get_conn() as conn:
                query = """
                    SELECT page_content, metadata FROM parent_chunks
                    WHERE parent_id = ?
                    AND (
                        (session_id = ? AND session_id != '')
                        OR (owner_id = ? AND persistent = 1)
                    )
                """
                cursor = conn.execute(query, (parent_id, session_id or "__NO_SESSION__", owner_id))
                row = cursor.fetchone()
                if row:
                    content, meta_str = row
                    try:
                        metadata = json.loads(meta_str)
                    except Exception:
                        metadata = {}
                    return Document(page_content=content, metadata=metadata)
        except Exception as exc:
            logger.error(f"Failed to query parent store for single ID {parent_id}: {exc}")

    return None


def delete_session_parents(session_id: str) -> None:
    """Delete parent documents belonging to a specific session."""
    if not session_id:
        return
    with _lock:
        try:
            with _get_conn() as conn:
                conn.execute("DELETE FROM parent_chunks WHERE session_id = ?", (session_id,))
                conn.commit()
            logger.info(f"Purged session {session_id} parent chunks from SQLite parent store.")
        except Exception as exc:
            logger.error(f"Failed to delete session {session_id} parents: {exc}")


def clear_parents() -> None:
    """Clear all parent document entries."""
    with _lock:
        try:
            with _get_conn() as conn:
                conn.execute("DELETE FROM parent_chunks")
                conn.commit()
            logger.info("Cleared all documents from SQLite parent store.")
        except Exception as exc:
            logger.error(f"Failed to clear parent store database: {exc}")
