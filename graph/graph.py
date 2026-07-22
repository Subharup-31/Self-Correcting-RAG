"""
Master LangGraph assembly for the Ultimate Self-Correcting RAG Pipeline.

Flow:
  Entry → route_question
    ├─ "direct_llm" → generate → END
    ├─ "websearch"  → web_search → retrieve → grade_documents → ...
    └─ "vectorstore" → (decompose) → retrieve (hybrid+HyDE) → grade_documents
                                                       │
                              ┌───────────────────────┼───────────────────────┐
                          "correct"              "ambiguous"            "incorrect"
                              │                      │                      │
                   detect_contradiction        clarify (ask Q)        query_rewrite
                              │                     END                       │
                       (no branch)                                  retrieve → grade
                              │                                  (max 3 retries)
                       rerank (cross-encoder)            (or web_search fallback)
                              │
                       generate (+ few-shot)
                              │
                       grade_hallucination
                       ┌──────┴──────┐
                  not grounded    grounded
                       │              │
                 re-generate   confidence_scorer
                 (max 3)            │
                              ┌─────┴─────┐
                          low_conf    high_conf
                              │           │
                         END (flag)  grade_answer
                                          │
                                   ┌──────┴──────┐
                              not useful    useful
                                   │           │
                              web_search     END
                              (fallback)

Every branch uses add_conditional_edges; retry_count / regen_count guard against
infinite loops (max 3 each).
"""

from __future__ import annotations

import time
import operator
from typing import Annotated, Optional

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from loguru import logger

from config import SelfCorrectionConfig
from graph.state import GraphState, initial_state
from graph.chains.few_shot_learner import get_few_shot_learner
from graph.chains.query_decomposer import decompose_query, is_complex_query

# Node functions
from graph.nodes.route import (
    route_question, route_decision,
    ROUTE, RETRIEVE, WEBSEARCH, DIRECT_LLM,
)
from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import (
    grade_documents, decide_after_grading, GRADE_DOCUMENTS,
    GO_DETECT_CONTRADICTION, GO_CLARIFY, GO_WEB_SEARCH,
)
from graph.nodes.detect_contradiction import (
    detect_contradiction, DETECT_CONTRADICTION,
)
from graph.nodes.clarify import clarify, CLARIFY
from graph.nodes.web_search import web_search
from graph.nodes.rerank import rerank, RERANK
from graph.nodes.generate import generate, GENERATE
from graph.nodes.grade_hallucination import (
    grade_hallucination, decide_after_hallucination, GRADE_HALLUCINATION,
    GO_REGENERATE, GO_CONFIDENCE,
)
from graph.nodes.confidence_scorer import (
    confidence_scorer, decide_after_confidence, CONFIDENCE_SCORER,
)
from graph.nodes.grade_answer import (
    grade_answer, decide_after_answer, GRADE_ANSWER,
    GO_END_USEFUL, GO_WEB_NOT_USEFUL,
)
from graph.nodes.query_rewrite import rewrite_query, QUERY_REWRITE

# Retrieval components (built once, shared across nodes)
from retrieval.bm25_retriever import get_bm25_retriever
from retrieval.vector_store import get_vector_store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.hyde import HyDERetriever

# Extra node names used in wiring
QUERY_DECOMPOSE = "query_decompose"
REGENERATE = "regenerate"
FEW_SHOT_INJECT = "few_shot_inject"


# ============================================================
# Retrieval component singletons
# ============================================================
_hybrid_retriever: Optional[HybridRetriever] = None
_hyde_retriever: Optional[HyDERetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(get_vector_store(), get_bm25_retriever())
    return _hybrid_retriever


def get_hyde_retriever() -> HyDERetriever:
    global _hyde_retriever
    if _hyde_retriever is None:
        _hyde_retriever = HyDERetriever(get_hybrid_retriever())
    return _hyde_retriever


# ============================================================
# Node wrappers (bind the shared retriever + handle edge cases)
# ============================================================
def _node_decompose(state: GraphState) -> GraphState:
    """Decompose complex queries before retrieval."""
    question = state["question"]
    if not is_complex_query(question):
        return {}
    subs = decompose_query(question)
    techniques = list(state.get("techniques_used", []))
    if subs and "Query Decomposition" not in techniques:
        techniques.append("Query Decomposition")
    return {"sub_questions": subs, "techniques_used": techniques}


def _node_retrieve(state: GraphState) -> GraphState:
    return retrieve(state, get_hybrid_retriever(), get_hyde_retriever(), use_hyde=True)


def _node_web_search(state: GraphState) -> GraphState:
    return web_search(state)


def _node_query_rewrite(state: GraphState) -> GraphState:
    """Rewrite then immediately re-retrieve."""
    new_state = rewrite_query(state)
    # Merge and re-retrieve with the rewritten query.
    merged = {**state, **new_state}
    retrieved = retrieve(merged, get_hybrid_retriever(), get_hyde_retriever(), use_hyde=True)
    # Merge new docs into existing.
    existing_docs = list(state.get("documents", []))
    new_docs = retrieved.get("documents", [])
    seen = {d.page_content[:200] for d in existing_docs}
    for d in new_docs:
        if d.page_content[:200] not in seen:
            existing_docs.append(d)
            seen.add(d.page_content[:200])
    return {
        "question": new_state.get("question", state["question"]),
        "retry_count": new_state.get("retry_count", state.get("retry_count", 0) + 1),
        "documents": existing_docs,
        "techniques_used": new_state.get("techniques_used", state.get("techniques_used", [])),
    }


def _node_few_shot_inject(state: GraphState) -> GraphState:
    """Pull similar past examples and stash the few-shot prefix in state.

    We store it as a transient key via a module-global so the generate node can
    read it without bloating GraphState's schema.
    """
    global _pending_few_shot_prefix
    try:
        prefix = get_few_shot_learner().get_dynamic_prompt(state["question"])
    except Exception:
        prefix = ""
    _pending_few_shot_prefix = prefix
    techniques = list(state.get("techniques_used", []))
    if prefix and "Dynamic Few-Shot" not in techniques:
        techniques.append("Dynamic Few-Shot")
    return {"techniques_used": techniques}


_pending_few_shot_prefix = ""


def _node_generate(state: GraphState) -> GraphState:
    global _pending_few_shot_prefix
    result = generate(state, few_shot_prefix=_pending_few_shot_prefix)
    _pending_few_shot_prefix = ""  # reset
    return result


def _node_regenerate(state: GraphState) -> GraphState:
    """Re-generate after hallucination failure. Bumps regen_count."""
    regen_count = state.get("regen_count", 0) + 1
    logger.info(f"---REGENERATE #{regen_count}---")
    # Force the generation to be more conservative by adding an instruction.
    new_state = {**state, "regen_count": regen_count}
    gen = generate(new_state, few_shot_prefix=_pending_few_shot_prefix)
    gen["regen_count"] = regen_count
    return gen


def _node_direct_llm(state: GraphState) -> GraphState:
    """Generate an answer with no retrieval (greetings, math, chit-chat)."""
    logger.info("---DIRECT LLM (no retrieval)---")
    # Provide empty context; the generation prompt will handle it.
    empty_state = {**state, "documents": []}
    return generate(empty_state)


def _node_end_with_sources(state: GraphState) -> GraphState:
    """Terminal node: build the sources list from documents."""
    sources = []
    seen = set()
    for d in state.get("documents", []):
        key = (d.metadata.get("source", ""), d.metadata.get("page_number", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page_number", "?"),
            "doc_type": d.metadata.get("doc_type", ""),
            "excerpt": d.page_content[:150].replace("\n", " ") + "...",
        })
    return {"sources": sources}


# ============================================================
# Build the graph
# ============================================================
def build_graph():
    """Construct and compile the master LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # ---- Add nodes ----
    workflow.add_node(ROUTE, route_question)
    workflow.add_node(QUERY_DECOMPOSE, _node_decompose)
    workflow.add_node(RETRIEVE, _node_retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(DETECT_CONTRADICTION, detect_contradiction)
    workflow.add_node(CLARIFY, clarify)
    workflow.add_node(QUERY_REWRITE, _node_query_rewrite)
    workflow.add_node(WEBSEARCH, _node_web_search)
    workflow.add_node(RERANK, rerank)
    workflow.add_node(FEW_SHOT_INJECT, _node_few_shot_inject)
    workflow.add_node(GENERATE, _node_generate)
    workflow.add_node(GRADE_HALLUCINATION, grade_hallucination)
    workflow.add_node(REGENERATE, _node_regenerate)
    workflow.add_node(CONFIDENCE_SCORER, confidence_scorer)
    workflow.add_node(GRADE_ANSWER, grade_answer)
    workflow.add_node(DIRECT_LLM, _node_direct_llm)
    workflow.add_node("finalize", _node_end_with_sources)

    # ---- Entry: route ----
    workflow.add_conditional_edges(
        START,
        # We route from the entry; but route_question needs to run first.
        # Simpler: START → ROUTE, then ROUTE conditional edges.
        lambda s: ROUTE,
        {ROUTE: ROUTE},
    )

    workflow.add_conditional_edges(
        ROUTE,
        route_decision,
        {
            RETRIEVE: QUERY_DECOMPOSE,
            WEBSEARCH: WEBSEARCH,
            DIRECT_LLM: DIRECT_LLM,
        },
    )

    # direct_llm → generate (shared) → END
    workflow.add_edge(DIRECT_LLM, GENERATE)
    # When direct_llm feeds generate, we want to skip hallucination/confidence
    # complexity; route straight to finalize.
    # We handle this by having GENERATE's conditional edge check route.

    # decompose → retrieve
    workflow.add_edge(QUERY_DECOMPOSE, RETRIEVE)

    # retrieve → grade_documents
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

    # grade_documents → {detect_contradiction | clarify | query_rewrite}
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_after_grading,
        {
            GO_DETECT_CONTRADICTION: DETECT_CONTRADICTION,
            GO_CLARIFY: CLARIFY,
            GO_WEB_SEARCH: QUERY_REWRITE,
        },
    )

    # clarify → END (we return the clarifying question to the user)
    workflow.add_edge(CLARIFY, "finalize")
    workflow.add_edge("finalize", END)

    # detect_contradiction → rerank (continue regardless; contradiction is surfaced)
    workflow.add_edge(DETECT_CONTRADICTION, RERANK)

    # query_rewrite (re-retrieve) → grade_documents (loop, guarded by retry_count)
    # If we've hit max retries, fall back to web search instead.
    def _after_rewrite(state: GraphState) -> str:
        if state.get("retry_count", 0) >= SelfCorrectionConfig.MAX_RETRY_COUNT:
            return WEBSEARCH
        return GRADE_DOCUMENTS

    workflow.add_conditional_edges(
        QUERY_REWRITE,
        _after_rewrite,
        {GRADE_DOCUMENTS: GRADE_DOCUMENTS, WEBSEARCH: WEBSEARCH},
    )

    # web_search → rerank (use web results + whatever survived)
    workflow.add_edge(WEBSEARCH, RERANK)

    # rerank → few_shot_inject → generate
    workflow.add_edge(RERANK, FEW_SHOT_INJECT)
    workflow.add_edge(FEW_SHOT_INJECT, GENERATE)

    # generate → grade_hallucination (but direct_llm path skips to finalize)
    def _after_generate(state: GraphState) -> str:
        # Direct-LLM answers skip the heavy grading.
        if state.get("route") == "direct_llm":
            return "finalize"
        return GRADE_HALLUCINATION

    workflow.add_conditional_edges(
        GENERATE,
        _after_generate,
        {GRADE_HALLUCINATION: GRADE_HALLUCINATION, "finalize": "finalize"},
    )

    # grade_hallucination → {regenerate | confidence}
    workflow.add_conditional_edges(
        GRADE_HALLUCINATION,
        decide_after_hallucination,
        {GO_REGENERATE: REGENERATE, GO_CONFIDENCE: CONFIDENCE_SCORER},
    )

    # regenerate → grade_hallucination (loop)
    workflow.add_edge(REGENERATE, GRADE_HALLUCINATION)

    # confidence_scorer → {end_flagged (finalize) | grade_answer}
    workflow.add_conditional_edges(
        CONFIDENCE_SCORER,
        decide_after_confidence,
        {"end_flagged": "finalize", "grade_answer": GRADE_ANSWER},
    )

    # grade_answer → {END | web_search fallback}
    def _after_answer(state: GraphState) -> str:
        decision = decide_after_answer(state)
        if decision == GO_WEB_NOT_USEFUL:
            return WEBSEARCH
        return "finalize"

    workflow.add_conditional_edges(
        GRADE_ANSWER,
        _after_answer,
        {WEBSEARCH: WEBSEARCH, "finalize": "finalize"},
    )

    app = workflow.compile()
    logger.info("Master LangGraph compiled successfully.")
    return app


# ============================================================
# Compiled graph singleton
# ============================================================
_app = None
_app_lock = __import__("threading").Lock()


def get_app():
    """Return the compiled graph (built once)."""
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                _app = build_graph()
    return _app


# ============================================================
# Public entry point
# ============================================================
def run_query(question: str, config: dict | None = None) -> dict:
    """Run a single question through the full pipeline. Returns the final state."""
    start = time.time()
    state = initial_state(question)
    app = get_app()
    logger.info(f"=== RUN QUERY: {question} ===")
    try:
        final_state = app.invoke(state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Graph execution failed: {exc}")
        final_state = {**state, "generation": f"[Pipeline error: {exc}]"}

    elapsed = time.time() - start
    final_state["processing_time"] = round(elapsed, 3)
    logger.info(
        f"=== QUERY DONE in {elapsed:.2f}s | "
        f"confidence={final_state.get('confidence_score', 0):.2f} "
        f"techniques={final_state.get('techniques_used', [])} ==="
    )
    return final_state


def stream_query(question: str, config: dict | None = None):
    """Stream intermediate states (for the SSE pipeline-trace endpoint).
    
    The final __done__ event now contains the complete final_state so the
    frontend does NOT need to issue a second POST /api/query to get the answer.
    """
    import time as _time
    start = _time.time()
    state = initial_state(question)
    app = get_app()
    final_state = state  # will be overwritten by the last node update
    all_updates: dict = {}

    for chunk in app.stream(state, config=config, stream_mode="updates"):
        # Each chunk is {node_name: state_update}
        for node_name, update in chunk.items():
            elapsed = round(_time.time() - start, 3)
            yield {"node": node_name, "update": _safe_update(update), "elapsed": elapsed}
            # Accumulate updates so we can reconstruct the final state
            all_updates.update(update)

    # Build the final state from accumulated updates
    elapsed_total = round(_time.time() - start, 3)
    final_state = {**state, **all_updates, "processing_time": elapsed_total}

    # Emit __done__ with the full final result embedded
    yield {
        "node": "__done__",
        "update": {},
        "elapsed": elapsed_total,
        "final_state": _safe_final_state(final_state),
    }


def _safe_update(update: dict) -> dict:
    """Make a state update JSON-serializable for streaming."""
    safe = {}
    for k, v in update.items():
        if isinstance(v, list) and v and hasattr(v[0], "page_content"):
            # Documents → summarize
            safe[k] = [{"content": d.page_content[:120], "metadata": d.metadata}
                       for d in v[:5]]
        else:
            try:
                import json
                json.dumps(v)
                safe[k] = v
            except Exception:
                safe[k] = str(v)
    return safe


def _safe_final_state(state: dict) -> dict:
    """Serialize the full final state into a JSON-friendly dict matching QueryResponse."""
    import json

    def _jsonable(v):
        if isinstance(v, list) and v and hasattr(v[0], "page_content"):
            return [
                {
                    "source": d.metadata.get("source", ""),
                    "page": d.metadata.get("page"),
                    "doc_type": d.metadata.get("doc_type"),
                    "excerpt": d.page_content[:200],
                    "rerank_score": d.metadata.get("rerank_score"),
                }
                for d in v
            ]
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    return {k: _jsonable(v) for k, v in state.items()}


if __name__ == "__main__":
    # Quick smoke test
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is a cross-encoder reranker?"
    result = run_query(q)
    print("\n=== RESULT ===")
    print(f"Answer: {result.get('generation', '')[:300]}")
    print(f"Confidence: {result.get('confidence_score', 0)}")
    print(f"Low confidence: {result.get('low_confidence', False)}")
    print(f"Techniques: {result.get('techniques_used', [])}")
