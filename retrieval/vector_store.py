"""
ChromaDB vector store backed by BAAI/bge-m3 embeddings.

- Persistent storage under ./chroma_db
- Indexes child chunks for retrieval (precise matching)
- Stores parent_id in each chunk's metadata so the graph can expand context
- Threads-safe singleton so the embedding model is loaded only once
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from config import CHROMA_PERSIST_DIR, ModelConfig, RetrievalConfig

COLLECTION_NAME = "ultimate_rag_children"


@lru_cache(maxsize=1)
def get_embeddings():
    """Singleton embeddings. Supports 'huggingface', 'google' (Gemini), 'openai', and 'nvidia' API providers."""
    provider = ModelConfig.EMBEDDING_PROVIDER
    logger.info(f"Initializing embedding provider: {provider}")

    if provider == "google":
        import os
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from config import APIKeys
        logger.info(f"Loading Google GenAI embeddings: {ModelConfig.EMBEDDING_MODEL}")
        return GoogleGenerativeAIEmbeddings(
            model=ModelConfig.EMBEDDING_MODEL,
            google_api_key=APIKeys.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY"),
        )
        
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Loading OpenAI embeddings: {ModelConfig.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(model=ModelConfig.EMBEDDING_MODEL)

    elif provider == "nvidia":
        import os
        from langchain_openai import OpenAIEmbeddings
        from config import APIKeys
        logger.info(f"Loading NVIDIA embeddings: {ModelConfig.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=ModelConfig.EMBEDDING_MODEL,
            openai_api_key=APIKeys.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1",
        )
        
    elif provider == "openrouter":
        import os
        from langchain_openai import OpenAIEmbeddings
        from config import APIKeys
        logger.info(f"Loading OpenRouter embeddings: {ModelConfig.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=ModelConfig.EMBEDDING_MODEL,
            openai_api_key=APIKeys.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
        )
        
    else:
        # Default: huggingface (local)
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Loading local HuggingFace embedding model: {ModelConfig.EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=ModelConfig.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


class VectorStore:
    """Persistent Qdrant (or ChromaDB-backed) vector store for child chunks."""

    def __init__(
        self,
        persist_dir: str = str(CHROMA_PERSIST_DIR),
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._embeddings = embeddings or get_embeddings()
        self._lock = threading.RLock()
        self._store = None
        # Will lazily connect on first use.

    @property
    def store(self):
        if self._store is None:
            with self._lock:
                if self._store is None:
                    from config import QdrantConfig
                    if QdrantConfig.ENDPOINT and QdrantConfig.API_KEY:
                        import time
                        from langchain_qdrant import QdrantVectorStore
                        from qdrant_client import QdrantClient
                        from qdrant_client.http.models import Distance, VectorParams

                        client = None
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                client = QdrantClient(
                                    url=QdrantConfig.ENDPOINT,
                                    api_key=QdrantConfig.API_KEY,
                                    timeout=30,
                                )
                                # Ensure collection exists
                                collections = client.get_collections().collections
                                collection_names = [c.name for c in collections]
                                if self.collection_name not in collection_names:
                                    client.create_collection(
                                        collection_name=self.collection_name,
                                        vectors_config=VectorParams(
                                            size=ModelConfig.EMBEDDING_DIM,
                                            distance=Distance.COSINE
                                        )
                                    )
                                    logger.info(f"Created Qdrant collection: {self.collection_name}")
                                else:
                                    try:
                                        coll_info = client.get_collection(self.collection_name)
                                        vectors_config = coll_info.config.params.vectors
                                        current_size = None
                                        if hasattr(vectors_config, "size"):
                                            current_size = vectors_config.size
                                        elif isinstance(vectors_config, dict):
                                            current_size = vectors_config.get("size")
                                        else:
                                            current_size = getattr(vectors_config, "size", None)
                                            
                                        if current_size and current_size != ModelConfig.EMBEDDING_DIM:
                                            logger.warning(
                                                f"Vector dimension mismatch in collection {self.collection_name}: "
                                                f"DB has {current_size}, config expects {ModelConfig.EMBEDDING_DIM}. "
                                                "Recreating collection to avoid API errors..."
                                            )
                                            client.delete_collection(self.collection_name)
                                            client.create_collection(
                                                collection_name=self.collection_name,
                                                vectors_config=VectorParams(
                                                    size=ModelConfig.EMBEDDING_DIM,
                                                    distance=Distance.COSINE
                                                )
                                            )
                                    except Exception as e:
                                        logger.warning(f"Failed to verify Qdrant collection dimensions: {e}")
                                break
                            except Exception as exc:
                                if attempt == max_retries - 1:
                                    logger.error(f"Failed to connect to Qdrant Cloud after {max_retries} attempts: {exc}")
                                    raise exc
                                logger.warning(f"Qdrant connection attempt {attempt+1} failed ({exc}). Retrying in 2s...")
                                time.sleep(2)

                        self._store = QdrantVectorStore(
                            client=client,
                            collection_name=self.collection_name,
                            embedding=self._embeddings,
                        )
                        logger.info(
                            f"QdrantCloudVectorStore opened: {QdrantConfig.ENDPOINT} "
                            f"(collection={self.collection_name})"
                        )
                    else:
                        from langchain_chroma import Chroma
                        self._store = Chroma(
                            collection_name=self.collection_name,
                            embedding_function=self._embeddings,
                            persist_directory=self.persist_dir,
                        )
                        logger.info(
                            f"ChromaDB opened: {self.persist_dir} "
                            f"(collection={self.collection_name})"
                        )
        return self._store

    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Document]) -> None:
        """Add (child) documents to the vector store.

        Generates stable ids from metadata so re-ingesting the same file does
        not create duplicate entries.
        """
        if not documents:
            return
        
        import time
        
        # Batch size for embedding requests to prevent hitting the rate limit
        batch_size = 15
        
        # We process in batches and handle rate limits with backoff
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = [self._make_id(d, i + idx) for idx, d in enumerate(batch_docs)]
            
            max_retries = 5
            backoff_delay = 5.0
            
            for attempt in range(max_retries):
                try:
                    with self._lock:
                        self.store.add_documents(documents=batch_docs, ids=batch_ids)
                    logger.info(f"Added batch of {len(batch_docs)} chunks (total: {i + len(batch_docs)}/{len(documents)})")
                    break
                except Exception as e:
                    err_str = str(e)
                    if "Collection" in err_str and "doesn't exist" in err_str:
                        logger.warning(
                            f"Collection {self.collection_name} does not exist. "
                            "Resetting connection to trigger auto-recreation..."
                        )
                        with self._lock:
                            self._store = None
                        # Referencing self.store will auto-recreate the collection
                        _ = self.store
                        continue  # retry this batch immediately
                    elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        logger.warning(
                            f"Rate limit hit during vector store ingestion: {e}. "
                            f"Retrying batch in {backoff_delay}s... (Attempt {attempt+1}/{max_retries})"
                        )
                        time.sleep(backoff_delay)
                        backoff_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Error adding batch to vector store: {e}")
                        raise e
            else:
                raise Exception("Failed to add documents to vector store after multiple retries due to rate limiting.")

    def search(self, query: str, k: int = RetrievalConfig.VECTOR_TOP_K) -> List[Document]:
        """Semantic similarity search."""
        if not query.strip():
            return []

        def _run_search():
            try:
                results = self.store.similarity_search_with_relevance_scores(query, k=k)
                return [d for d, _ in results]
            except Exception:
                # Fallback if relevance scores not supported
                return self.store.similarity_search(query, k=k)

        try:
            return _run_search()
        except Exception as e:
            err_str = str(e)
            if "Collection" in err_str and "doesn't exist" in err_str:
                logger.warning(
                    f"Collection {self.collection_name} does not exist. "
                    "Resetting connection to trigger auto-recreation..."
                )
                with self._lock:
                    self._store = None
                return _run_search()
            raise e

    def search_with_scores(self, query: str, k: int = RetrievalConfig.VECTOR_TOP_K):
        """Return (Document, score) tuples sorted by similarity."""
        def _run_search_with_scores():
            try:
                results = self.store.similarity_search_with_score(query, k=k)
                # Normalize depending on vector store response (Qdrant uses cosine/dot, Chroma uses distance)
                from langchain_qdrant import QdrantVectorStore
                if isinstance(self.store, QdrantVectorStore):
                    return [(d, float(s)) for d, s in results]
                return [(d, 1.0 - (s / 2.0)) for d, s in results]
            except Exception:
                docs = self.store.similarity_search(query, k=k)
                return [(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]

        try:
            return _run_search_with_scores()
        except Exception as e:
            err_str = str(e)
            if "Collection" in err_str and "doesn't exist" in err_str:
                logger.warning(
                    f"Collection {self.collection_name} does not exist. "
                    "Resetting connection to trigger auto-recreation..."
                )
                with self._lock:
                    self._store = None
                return _run_search_with_scores()
            raise e

    def as_retriever(self, k: int = RetrievalConfig.VECTOR_TOP_K):
        return self.store.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        def _run_count():
            from langchain_qdrant import QdrantVectorStore
            if isinstance(self.store, QdrantVectorStore):
                result = self.store.client.count(
                    collection_name=self.collection_name,
                    exact=True
                )
                return result.count
            return self.store._collection.count()

        try:
            return _run_count()
        except Exception as e:
            err_str = str(e)
            if "Collection" in err_str and "doesn't exist" in err_str:
                logger.warning(
                    f"Collection {self.collection_name} does not exist. "
                    "Resetting connection to trigger auto-recreation..."
                )
                with self._lock:
                    self._store = None
                try:
                    return _run_count()
                except Exception:
                    return 0
            return 0

    def clear(self) -> None:
        """Delete the collection entirely (used by API DELETE /documents)."""
        with self._lock:
            try:
                from langchain_qdrant import QdrantVectorStore
                if isinstance(self.store, QdrantVectorStore):
                    self.store.client.delete_collection(self.collection_name)
                    # Recreate immediately to prevent 404s in other running workers/processes
                    from qdrant_client.http.models import Distance, VectorParams
                    self.store.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=ModelConfig.EMBEDDING_DIM,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Recreated Qdrant collection after clear: {self.collection_name}")
                else:
                    self.store.delete_collection()
            except Exception as e:
                logger.warning(f"Error clearing vector store: {e}")
            self._store = None  # force re-init on next access
        logger.info("Vector store cleared.")


    @staticmethod
    def _make_id(doc: Document, index: int) -> str:
        """Stable id: source::page::parent_id::child_index::index (converted to UUIDv5 for Qdrant)"""
        import uuid
        m = doc.metadata
        stable_string = ":".join([
            str(m.get("source", "unknown")),
            str(m.get("page_number", 0)),
            str(m.get("parent_id", "noparent")),
            str(m.get("child_index", index)),
            str(index),
        ])
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, stable_string))



# Module-level singleton for convenience
_vector_store: Optional[VectorStore] = None
_vs_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        with _vs_lock:
            if _vector_store is None:
                _vector_store = VectorStore()
    return _vector_store
