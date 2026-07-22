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
    MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    GRADER_TEMPERATURE = 0.0          # Graders must be deterministic
    GENERATION_TEMPERATURE = 0.3      # Generation can be slightly creative
    HYDE_TEMPERATURE = 0.7            # HyDE benefits from creativity


# ============================================================
# Embeddings & Models
# ============================================================
class ModelConfig:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM = 1024
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K = 5


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
# API keys (resolved at import)
# ============================================================
class APIKeys:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_API_KEY_FALLBACK = os.getenv("GOOGLE_API_KEY_FALLBACK", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


# ============================================================
# Qdrant Configuration
# ============================================================
class QdrantConfig:
    ENDPOINT = os.getenv("QDRANT_ENDPOINT", "")
    API_KEY = os.getenv("QDRANT_API_KEY", "")


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

