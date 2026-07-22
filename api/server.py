"""
FastAPI server for the Ultimate Self-Correcting RAG Pipeline.

Endpoints:
  POST   /api/upload         Upload documents (PDF, images, Word, txt)
  POST   /api/query          Run a query through the full pipeline
  POST   /api/feedback       Submit thumbs up/down for few-shot learning
  GET    /api/statistics     System performance + store stats
  GET    /api/evaluate       Run the 12-question evaluation harness
  GET    /api/health         Health check
  DELETE /api/documents      Clear all indexed documents
  GET    /api/query/stream   SSE stream with step-by-step pipeline trace

Run:  uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import APIKeys, DOCUMENTS_DIR, ServerConfig

# ---------------------------------------------------------------- #
# App setup
# ---------------------------------------------------------------- #
app = FastAPI(
    title="Ultimate Self-Correcting RAG",
    description="A RAG pipeline that detects insufficient/contradictory context "
                "and self-corrects instead of hallucinating.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- #
# Request / response models
# ---------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    stream: bool = Field(False, description="If true, use the SSE endpoint instead")


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    is_positive: bool = True
    feedback_score: float = Field(1.0, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence_score: float
    low_confidence: bool
    clarification_needed: bool
    clarification_question: Optional[str] = None
    contradiction_found: bool
    contradiction_detail: Optional[str] = None
    crag_state: str
    hallucination_free: bool
    web_search_used: bool
    sources: List[dict]
    techniques_used: List[str]
    processing_time: float
    retry_count: int


def _state_to_response(state: dict) -> QueryResponse:
    """Convert a final GraphState into an API response."""
    return QueryResponse(
        query=state.get("question", ""),
        answer=state.get("generation", ""),
        confidence_score=float(state.get("confidence_score", 0.0)),
        low_confidence=bool(state.get("low_confidence", False)),
        clarification_needed=bool(state.get("clarification_needed", False)),
        clarification_question=state.get("clarification_question") or None,
        contradiction_found=bool(state.get("contradiction_found", False)),
        contradiction_detail=state.get("contradiction_detail") or None,
        crag_state=state.get("crag_state", ""),
        hallucination_free=bool(state.get("hallucination_free", False)),
        web_search_used=bool(state.get("web_search_used", False)),
        sources=state.get("sources", []),
        techniques_used=state.get("techniques_used", []),
        processing_time=float(state.get("processing_time", 0.0)),
        retry_count=int(state.get("retry_count", 0)),
    )


# ---------------------------------------------------------------- #
# Lazy import of heavy pipeline (so /health works without models loaded)
# ---------------------------------------------------------------- #
def _run_query(question: str) -> dict:
    from graph.graph import run_query
    return run_query(question)


def _stream_query(question: str):
    from graph.graph import stream_query
    yield from stream_query(question)


# ---------------------------------------------------------------- #
# Endpoints
# ---------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    """Health check — reports API key presence + store readiness."""
    return {
        "status": "ok",
        "google_api_key": bool(APIKeys.GOOGLE_API_KEY),
        "openai_api_key": bool(APIKeys.OPENAI_API_KEY),
        "tavily_api_key": bool(APIKeys.TAVILY_API_KEY),
        "documents_dir": str(DOCUMENTS_DIR),
        "documents_dir_exists": DOCUMENTS_DIR.exists(),
    }


@app.get("/")
async def root():
    return {
        "name": "Ultimate Self-Correcting RAG",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /api/upload", "POST /api/query", "POST /api/feedback",
            "GET /api/statistics", "GET /api/evaluate", "GET /api/health",
            "DELETE /api/documents", "GET /api/query/stream",
        ],
    }


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest one or more documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        # Save to documents/
        safe_name = f"{uuid.uuid4().hex[:8]}_{Path(f.filename).name}"
        dest = DOCUMENTS_DIR / safe_name
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
        except Exception as exc:
            logger.warning(f"Failed to save {f.filename}: {exc}")
            results.append({"file": f.filename, "status": "save_failed", "error": str(exc)})
            continue

        # Ingest
        try:
            from ingestion.pipeline import ingest_file
            summary = await asyncio.get_event_loop().run_in_executor(
                None, ingest_file, str(dest)
            )
            results.append({
                "file": f.filename,
                "saved_as": safe_name,
                "status": "ingested",
                "summary": summary,
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Ingest failed for {f.filename}")
            results.append({
                "file": f.filename, "status": "ingest_failed", "error": str(exc)
            })
    return {"results": results}


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run a query through the full self-correcting pipeline."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        state = await asyncio.get_event_loop().run_in_executor(
            None, _run_query, req.query
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return _state_to_response(state)


@app.get("/api/query/stream")
async def query_stream(request: Request, q: str = ""):
    """SSE stream of the pipeline trace for a query."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Missing 'q' query parameter")

    async def event_generator():
        def run_sync():
            for event in _stream_query(q):
                return_event = event
                # Run in executor to avoid blocking
                return return_event
        # Stream from the sync generator
        loop = asyncio.get_event_loop()
        try:
            for event in _stream_query(q):
                if await request.is_disconnected():
                    break
                payload = json.dumps(event, default=str)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'node': '__complete__'})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'node': '__error__', 'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """Store positive feedback as a few-shot example for future queries."""
    if req.is_positive:
        from graph.chains.few_shot_learner import get_few_shot_learner
        learner = get_few_shot_learner()
        learner.add_good_example(req.query, req.answer, req.feedback_score)
        stats = learner.get_stats()
        return {"status": "recorded", "total_examples": stats["total_examples"]}
    return {"status": "ignored_negative_feedback"}


@app.get("/api/statistics")
async def statistics():
    """Return store sizes + few-shot learner stats."""
    try:
        from ingestion.pipeline import get_ingestion_stats
        from graph.chains.few_shot_learner import get_few_shot_learner
        store_stats = get_ingestion_stats()
        learner_stats = get_few_shot_learner().get_stats()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Stats failed: {exc}")
        store_stats, learner_stats = {"error": str(exc)}, {"total_examples": 0}
    return {"stores": store_stats, "few_shot": learner_stats}


@app.get("/api/evaluate")
async def evaluate():
    """Run the full evaluation harness (baseline vs ultimate)."""
    from evaluation.harness import run_evaluation
    # Run in executor to avoid blocking the event loop (this is slow).
    result = await asyncio.get_event_loop().run_in_executor(None, run_evaluation)
    # Return only the metrics + summary, not the full per-question dumps (too big).
    return {
        "timestamp": result.get("timestamp"),
        "n_questions": result.get("n_questions"),
        "baseline_metrics": result.get("baseline_metrics"),
        "ultimate_metrics": result.get("ultimate_metrics"),
    }


@app.delete("/api/documents")
async def clear_documents():
    """Clear all indexed documents from both stores."""
    from ingestion.pipeline import reset_stores
    reset_stores()
    return {"status": "cleared"}


# ---------------------------------------------------------------- #
# Entrypoint
# ---------------------------------------------------------------- #
def serve(host: str = None, port: int = None):
    """Start the uvicorn server (called from main.py)."""
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=host or ServerConfig.HOST,
        port=port or ServerConfig.PORT,
        reload=False,
    )


if __name__ == "__main__":
    serve()
