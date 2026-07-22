"""
Parent-child chunking strategy for retrieval.

Why parent-child?
  - Child chunks (small, 256 tokens) are embedded and retrieved. Their small
    size gives precise semantic matching.
  - Parent chunks (large, 1024 tokens) are stored alongside and returned as the
    *context* at generation time. This gives the LLM broader, more coherent
    context than the tiny retrieved child alone.

Each child chunk records its `parent_id` in metadata so the graph can expand
to the full parent at generation time.
"""

from __future__ import annotations

import uuid
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import RetrievalConfig


class ParentChildChunker:
    """Produces (child_chunks, parent_chunks) from a list of Documents."""

    def __init__(
        self,
        child_chunk_size: int = RetrievalConfig.CHUNK_SIZE * 4,  # tokens→chars (~4 chars/token)
        child_chunk_overlap: int = RetrievalConfig.CHUNK_OVERLAP * 4,
        parent_chunk_size: int = RetrievalConfig.PARENT_CHUNK_SIZE * 4,
    ):
        # Character-based splitters (approx 4 chars/token for English text).
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.debug(
            f"ParentChildChunker init: child={child_chunk_size}c, "
            f"parent={parent_chunk_size}c"
        )

    def chunk(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """Split documents into parent and child chunks.

        Returns:
            (child_chunks, parent_chunks)
            - parent_chunks: large context blocks, each with a stable `parent_id`.
            - child_chunks: small retrieval blocks, each carrying its parent's id.
        """
        parent_chunks: List[Document] = []
        child_chunks: List[Document] = []

        for doc in documents:
            # Step 1: split into parents.
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent.metadata = {
                    **doc.metadata,
                    "parent_id": parent_id,
                    "chunk_role": "parent",
                }
                parent_chunks.append(parent)

                # Step 2: split each parent into children.
                children = self.child_splitter.split_documents([parent])
                for idx, child in enumerate(children):
                    child.metadata = {
                        **parent.metadata,
                        "parent_id": parent_id,
                        "chunk_role": "child",
                        "child_index": idx,
                    }
                    child_chunks.append(child)

        logger.info(
            f"Chunked {len(documents)} document(s) → "
            f"{len(parent_chunks)} parent(s), {len(child_chunks)} child(ren)"
        )
        return child_chunks, parent_chunks

    @staticmethod
    def get_parent(child: Document, parent_store: List[Document]) -> Document:
        """Look up the parent chunk for a given child by parent_id."""
        pid = child.metadata.get("parent_id")
        if pid is None:
            return child
        for p in parent_store:
            if p.metadata.get("parent_id") == pid:
                return p
        return child  # graceful fallback

    @staticmethod
    def expand_to_parents(
        children: List[Document], parent_store: List[Document]
    ) -> List[Document]:
        """Given retrieved child chunks, return the deduplicated parent chunks.

        Order is preserved by first appearance of each parent among the children.
        """
        seen = set()
        expanded: List[Document] = []
        for child in children:
            parent = ParentChildChunker.get_parent(child, parent_store)
            pid = parent.metadata.get("parent_id", id(parent))
            if pid not in seen:
                seen.add(pid)
                expanded.append(parent)
        return expanded


def chunk_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    """Convenience wrapper around ParentChildChunker.chunk."""
    return ParentChildChunker().chunk(documents)
