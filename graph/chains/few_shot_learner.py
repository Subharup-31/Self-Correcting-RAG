"""
Dynamic few-shot learning manager.

Ported from Self-Healing-RAG/backend/dynamic_prompting.py. Rewritten to use
LangChain + ChromaDB (instead of LlamaIndex) for storing/retrieving successful
Q&A pairs.

Functionality:
  - add_good_example(query, answer, feedback_score): store a successful Q&A pair
  - get_dynamic_prompt(current_query): retrieve similar past successes and
    format them as a few-shot prefix for the generation prompt
  - export_examples / import_examples: JSON persistence
  - get_stats(): summary of the learned store

Replaces the original "rebuild index on every add" with incremental ChromaDB
inserts, fixing the O(n) per-add cost noted in the source.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from config import FEW_SHOT_STORE
from llm import get_grader_llm

FEWSHOT_COLLECTION = "few_shot_examples"


class FewShotExample(BaseModel):
    """A stored successful Q&A pair."""
    query: str
    answer: str
    feedback_score: float = Field(ge=0.0, le=1.0)
    timestamp: str = ""
    example_id: str = ""

    def to_text(self) -> str:
        return f"Question: {self.query}\nAnswer: {self.answer}"


class FewShotLearner:
    """Stores successful Q&A pairs and retrieves similar ones for few-shot prompting."""

    def __init__(self, persist_path: str = str(FEW_SHOT_STORE), top_k: int = 2):
        self.persist_path = persist_path
        self.top_k = top_k
        self._lock = threading.RLock()
        self._examples: List[FewShotExample] = []
        self._vector_store = None
        self._load()

    # ------------------------------------------------------------------ #
    # Vector store (lazy)
    # ------------------------------------------------------------------ #
    def _get_vector_store(self):
        """Lazily build a Qdrant or ChromaDB store for few-shot examples."""
        if self._vector_store is None:
            import uuid
            from retrieval.vector_store import get_embeddings
            from config import QdrantConfig, ModelConfig

            if QdrantConfig.ENDPOINT and QdrantConfig.API_KEY:
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams

                client = QdrantClient(
                    url=QdrantConfig.ENDPOINT,
                    api_key=QdrantConfig.API_KEY,
                )
                try:
                    collections = client.get_collections().collections
                    collection_names = [c.name for c in collections]
                    if FEWSHOT_COLLECTION not in collection_names:
                        client.create_collection(
                            collection_name=FEWSHOT_COLLECTION,
                            vectors_config=VectorParams(
                                size=ModelConfig.EMBEDDING_DIM,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(f"Created Qdrant collection: {FEWSHOT_COLLECTION}")
                except Exception as exc:
                    logger.warning(f"Error checking/creating Qdrant few-shot collection: {exc}")

                self._vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=FEWSHOT_COLLECTION,
                    embedding=get_embeddings(),
                )
                logger.info(f"QdrantCloudVectorStore opened for few-shot examples")
            else:
                from langchain_chroma import Chroma
                self._vector_store = Chroma(
                    collection_name=FEWSHOT_COLLECTION,
                    embedding_function=get_embeddings(),
                    persist_directory=str(FEW_SHOT_STORE.parent / "few_shot_chroma"),
                )
                logger.info(f"ChromaDB opened for few-shot examples")
        return self._vector_store


    # ------------------------------------------------------------------ #
    # Persistence (JSON sidecar — keeps examples portable & inspectable)
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            import pathlib
            p = pathlib.Path(self.persist_path)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self._examples = [FewShotExample(**e) for e in data]
                logger.info(f"Loaded {len(self._examples)} few-shot examples from {p}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load few-shot store: {exc}")
            self._examples = []

    def _save(self) -> None:
        try:
            import pathlib
            p = pathlib.Path(self.persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            data = [e.model_dump() for e in self._examples]
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not save few-shot store: {exc}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_good_example(self, query: str, answer: str,
                         feedback_score: float = 1.0) -> FewShotExample:
        """Store a successful Q&A pair and add it to the retrieval index."""
        import uuid

        example = FewShotExample(
            query=query.strip(),
            answer=answer.strip(),
            feedback_score=max(0.0, min(1.0, feedback_score)),
            timestamp=datetime.now(timezone.utc).isoformat(),
            example_id=str(uuid.uuid4()),
        )
        with self._lock:
            self._examples.append(example)
            # Incremental insert into ChromaDB.
            try:
                self._get_vector_store().add_texts(
                    texts=[example.to_text()],
                    metadatas=[example.model_dump()],
                    ids=[example.example_id],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Few-shot vector insert failed: {exc}")
            self._save()
        logger.info(f"Added few-shot example (total: {len(self._examples)})")
        return example

    def get_similar_examples(self, current_query: str) -> List[FewShotExample]:
        """Retrieve the most similar past successful examples."""
        if not self._examples:
            return []
        try:
            docs = self._get_vector_store().similarity_search(current_query, k=self.top_k)
            examples = []
            for d in docs:
                meta = d.metadata or {}
                examples.append(FewShotExample(
                    query=meta.get("query", ""),
                    answer=meta.get("answer", d.page_content),
                    feedback_score=meta.get("feedback_score", 1.0),
                    timestamp=meta.get("timestamp", ""),
                    example_id=meta.get("example_id", ""),
                ))
            return examples
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Few-shot retrieval failed: {exc}")
            return []

    def get_dynamic_prompt(self, current_query: str) -> str:
        """Build a few-shot prefix string from similar past successes.

        Returns an empty string if no examples are available (so the caller can
        always concatenate it).
        """
        examples = self.get_similar_examples(current_query)
        if not examples:
            return ""
        blocks = []
        for i, ex in enumerate(examples, start=1):
            blocks.append(
                f"Example {i} (feedback score {ex.feedback_score:.1f}):\n"
                f"  Q: {ex.query}\n  A: {ex.answer}"
            )
        prefix = (
            f"Here are {len(examples)} example(s) of how to answer similar "
            f"questions correctly. Use them as guidance for tone and structure:\n\n"
            + "\n\n".join(blocks)
        )
        logger.info(f"Built dynamic few-shot prompt with {len(examples)} example(s)")
        return prefix

    def export_examples(self, filepath: str) -> None:
        """Export all examples to a JSON file."""
        out = pathlib.Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([e.model_dump() for e in self._examples], indent=2,
                       ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Exported {len(self._examples)} examples to {filepath}")

    def import_examples(self, filepath: str) -> int:
        """Import examples from a JSON file. Returns count imported."""
        try:
            data = json.loads(pathlib.Path(filepath).read_text(encoding="utf-8"))
            count = 0
            for e in data:
                try:
                    ex = FewShotExample(**e)
                    self._examples.append(ex)
                    try:
                        self._get_vector_store().add_texts(
                            texts=[ex.to_text()],
                            metadatas=[ex.model_dump()],
                            ids=[ex.example_id],
                        )
                    except Exception:
                        pass
                    count += 1
                except Exception:
                    continue
            self._save()
            logger.info(f"Imported {count} examples from {filepath}")
            return count
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Import failed: {exc}")
            return 0

    def get_stats(self) -> dict:
        if not self._examples:
            return {"total_examples": 0, "avg_feedback_score": 0.0}
        scores = [e.feedback_score for e in self._examples]
        return {
            "total_examples": len(self._examples),
            "avg_feedback_score": sum(scores) / len(scores),
            "sample_queries": [e.query for e in self._examples[-5:]],
        }


# Module-level singleton
_learner: Optional[FewShotLearner] = None
_learner_lock = threading.Lock()


def get_few_shot_learner() -> FewShotLearner:
    global _learner
    if _learner is None:
        with _learner_lock:
            if _learner is None:
                _learner = FewShotLearner()
    return _learner


# Make pathlib importable at module level for export/import methods
import pathlib  # noqa: E402
