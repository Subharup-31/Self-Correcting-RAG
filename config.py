"""
Central configuration for the Ultimate Self-Correcting RAG Pipeline.

All tunable constants live here. Values are read from environment variables
(with sensible defaults) so the system is configurable without code edits.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (idempotent)
load_dotenv()

# Only apply custom socket monkeypatch if explicitly enabled via environment variable
if os.getenv("ENABLE_SOCKET_MONKEYPATCH", "false").lower() == "true":
    import socket
    _orig_getaddrinfo = socket.getaddrinfo

    def _custom_getaddrinfo(host, port, *args, **kwargs):
        if host == "197cde2c-b60c-4e50-8623-54c2f729cddb.eu-west-1-0.aws.cloud.qdrant.io":
            return _orig_getaddrinfo("54.78.151.125", port, *args, **kwargs)
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = _custom_getaddrinfo

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_PERSIST_DIR = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").lstrip("./")
FEW_SHOT_STORE = BASE_DIR / "few_shot_examples.json"
EVALUATION_RESULTS = BASE_DIR / "evaluation" / "results.json"

# Ensure dirs exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LLM Configuration
# ============================================================
class LLMConfig:
    PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    _env_model = os.getenv("LLM_MODEL", "")
    if _env_model and _env_model != "gemini-1.5-flash":
        MODEL = _env_model
    else:
        _default_model = "gemini-3.5-flash-lite"
        if PROVIDER == "nvidia":
            _default_model = "meta/llama-3.1-70b-instruct"
        elif PROVIDER == "openai":
            _default_model = "gpt-4o-mini"
        elif PROVIDER == "openrouter":
            _default_model = "meta-llama/llama-3-70b-instruct"
        MODEL = _default_model
        
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    GRADER_TEMPERATURE = 0.0          # Graders must be deterministic
    GENERATION_TEMPERATURE = 0.3      # Generation can be slightly creative
    HYDE_TEMPERATURE = 0.7            # HyDE benefits from creativity


# ============================================================
# Embeddings & Models
# ============================================================
class ModelConfig:
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    
    _env_emb_model = os.getenv("EMBEDDING_MODEL", "")
    if _env_emb_model and _env_emb_model != "BAAI/bge-m3":
        EMBEDDING_MODEL = _env_emb_model
    else:
        if EMBEDDING_PROVIDER == "google":
            EMBEDDING_MODEL = "gemini-embedding-2"
        elif EMBEDDING_PROVIDER == "openai":
            EMBEDDING_MODEL = "text-embedding-3-large"
        elif EMBEDDING_PROVIDER == "nvidia":
            EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
        elif EMBEDDING_PROVIDER == "openrouter":
            EMBEDDING_MODEL = "openai/text-embedding-3-large"
        else:
            EMBEDDING_MODEL = "BAAI/bge-m3"

    _default_dim = 1024
    if EMBEDDING_PROVIDER == "google":
        if "gemini-embedding" in EMBEDDING_MODEL:
            _default_dim = 3072
        else:
            _default_dim = 768
    elif EMBEDDING_PROVIDER == "openai" or EMBEDDING_PROVIDER == "openrouter":
        if "text-embedding-3-small" in EMBEDDING_MODEL:
            _default_dim = 1536
        elif "text-embedding-ada-002" in EMBEDDING_MODEL:
            _default_dim = 1536
        else:
            _default_dim = 3072
    elif EMBEDDING_PROVIDER == "nvidia":
        _default_dim = 1024
    else:
        if "bge-m3" in EMBEDDING_MODEL:
            _default_dim = 1024
        elif "all-MiniLM-L6-v2" in EMBEDDING_MODEL:
            _default_dim = 384

    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", str(_default_dim)))
    
    RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "huggingface").lower()
    
    _env_rerank_model = os.getenv("RERANKER_MODEL", "")
    if _env_rerank_model and _env_rerank_model != "cross-encoder/ms-marco-MiniLM-L-6-v2":
        RERANKER_MODEL = _env_rerank_model
    else:
        if RERANKER_PROVIDER == "nvidia":
            RERANKER_MODEL = "nvidia/rerank-qa-mistral-4b"
        elif RERANKER_PROVIDER == "openrouter":
            RERANKER_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
        else:
            RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))


# ============================================================
# Retrieval
# ============================================================
class RetrievalConfig:
    CHUNK_SIZE = 256                  # child chunk size (tokens)
    CHUNK_OVERLAP = 64
    PARENT_CHUNK_SIZE = 1024
    VECTOR_TOP_K = 10
    BM25_TOP_K = 10
    HYBRID_FINAL_K = 6
    RRF_K = 60                        # Reciprocal Rank Fusion constant


# ============================================================
# Self-Correction
# ============================================================
class SelfCorrectionConfig:
    MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    CRAG_INITIAL_DOC_LIMIT = 5
    WEB_SEARCH_MAX_RESULTS = 3

    # Confidence score weights (must sum to 1.0)
    WEIGHT_HALLUCINATION = 0.40
    WEIGHT_CRAG = 0.25
    WEIGHT_ANSWER = 0.20
    WEIGHT_RETRY = 0.15

    CRAG_STATE_SCORES = {
        "correct": 1.0,
        "ambiguous": 0.5,
        "incorrect": 0.0,
    }


# ============================================================
# Dynamic API keys getter
# ============================================================
class _DynamicKeysMeta(type):
    @property
    def GOOGLE_API_KEY(cls) -> str:
        return os.getenv("GOOGLE_API_KEY", "")

    @property
    def GOOGLE_API_KEY_FALLBACK(cls) -> str:
        return os.getenv("GOOGLE_API_KEY_FALLBACK", "")

    @property
    def OPENAI_API_KEY(cls) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def TAVILY_API_KEY(cls) -> str:
        return os.getenv("TAVILY_API_KEY", "")

    @property
    def NVIDIA_API_KEY(cls) -> str:
        return os.getenv("NVIDIA_API_KEY", "")

    @property
    def OPENROUTER_API_KEY(cls) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")


class APIKeys(metaclass=_DynamicKeysMeta):
    pass


# ============================================================
# Dynamic Qdrant Configuration
# ============================================================
class _DynamicQdrantMeta(type):
    @property
    def ENDPOINT(cls) -> str:
        return os.getenv("QDRANT_ENDPOINT", "")

    @property
    def API_KEY(cls) -> str:
        return os.getenv("QDRANT_API_KEY", "")


class QdrantConfig(metaclass=_DynamicQdrantMeta):
    pass


# ============================================================
# API server
# ============================================================
class ServerConfig:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS = ["*"]


__all__ = [
    "BASE_DIR", "DOCUMENTS_DIR", "CHROMA_PERSIST_DIR",
    "FEW_SHOT_STORE", "EVALUATION_RESULTS",
    "LLMConfig", "ModelConfig", "RetrievalConfig",
    "SelfCorrectionConfig", "APIKeys", "ServerConfig",
    "QdrantConfig",
]

