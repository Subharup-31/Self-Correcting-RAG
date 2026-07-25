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

from config import SelfCorrectionConfig, APIKeys
from graph.state import GraphState, initial_state
from graph.chains.few_shot_learner import get_few_shot_learner
from graph.chains.query_decomposer import decompose_query, is_complex_query, multi_query_expansion

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
# Comparison query detection (for parallel web augmentation)
# ============================================================
_COMPARISON_TRIGGERS = (
    " vs ", " versus ", " compare ", " comparison ",
    " different from ", " differ from ", " unlike ",
    " compared to ", " relative to ", " better than ",
    " worse than ", " how does ", " what is the difference ",
)


def _is_comparison_query(question: str) -> bool:
    """Return True if the query asks to compare internal docs with external knowledge.

    Used to trigger parallel web augmentation alongside vector-store retrieval,
    so the LLM can synthesise differences across both sources in one shot.
    """
    q = f" {question.lower()} "
    return any(trigger in q for trigger in _COMPARISON_TRIGGERS)


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
    """Retrieve from the vector store, then optionally augment with web results
    when the query is asking to *compare* internal documents with external sources
    (e.g. "How does our leave policy differ from Google's?").

    Sub-question retrieval is already handled inside retrieve() itself.
    Multi-query expansion runs here in parallel for broader recall.
    """
    question = state["question"]
    use_hyde = is_complex_query(question)

    # ── Primary retrieval (vector store, BM25, HyDE, sub-questions) ──────────
    base_result = retrieve(state, get_hybrid_retriever(), get_hyde_retriever(), use_hyde=use_hyde)
    all_docs = list(base_result.get("documents", []))
    techniques = list(base_result.get("techniques_used", state.get("techniques_used", [])))
    seen_keys = {d.page_content[:200] for d in all_docs}
    owner_id = state.get("owner_id", "default_owner")
    session_id = state.get("session_id", "")

    # ── Multi-query expansion: 3 rephrased variations searched in parallel ──────
    # Runs for every query (not just complex) to improve recall diversity.
    # Wrapped in try/except so any LLM timeout doesn't block the main retrieval.
    try:
        variations = multi_query_expansion(question)
        if variations:
            import concurrent.futures

            def _fetch_variation(v: str):
                try:
                    return get_hybrid_retriever().retrieve(v, top_k=3,
                                                          owner_id=owner_id,
                                                          session_id=session_id)
                except Exception:
                    return []

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                var_results = list(ex.map(_fetch_variation, variations))

            added_var = 0
            for docs in var_results:
                for doc in docs:
                    key = doc.page_content[:200]
                    if key not in seen_keys:
                        all_docs.append(doc)
                        seen_keys.add(key)
                        added_var += 1

            if added_var:
                logger.info(f"[Retrieve] Multi-query expansion added {added_var} unique doc(s).")
                if "Multi-Query Expansion" not in techniques:
                    techniques.append("Multi-Query Expansion")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[Retrieve] Multi-query expansion failed (non-fatal): {exc}")

    # ── Comparison-query web augmentation ────────────────────────────────────
    # Only fires when:
    #   - the router chose vectorstore (we're in this node at all)
    #   - the query is a comparison ("X vs Y", "how does A differ from B")
    #   - Tavily key is configured
    if _is_comparison_query(question) and APIKeys.TAVILY_API_KEY:
        logger.info("[Retrieve] Comparison query detected — augmenting with web search.")
        try:
            # ── Targeted web query: only search for sub-queries that had NO local results ──
            # This avoids sending the full comparison question to Tavily (which returns noisy,
            # general results) and instead targets only the parts genuinely missing from the
            # local vector store.
            #
            # Safety: falls back to original full question if:
            #   - No sub-questions were decomposed (simple comparison query)
            #   - ALL sub-questions already had local results (nothing missing externally)
            sub_questions = state.get("sub_questions", [])
            web_query = question  # default: original behaviour preserved

            if sub_questions:
                _hr = get_hybrid_retriever()
                missing_subs = []
                for sq in sub_questions:
                    try:
                        sq_docs = _hr.retrieve(sq, top_k=1, owner_id=owner_id, session_id=session_id)
                        if not sq_docs:
                            missing_subs.append(sq)
                    except Exception:  # noqa: BLE001
                        # If retrieval check fails for a sub-query, conservatively treat it
                        # as missing so the web can fill the gap.
                        missing_subs.append(sq)

                if missing_subs:
                    web_query = " ".join(missing_subs)
                    logger.info(
                        f"[Retrieve] Targeting web search at {len(missing_subs)} sub-query(ies) "
                        f"not found locally: {missing_subs}"
                    )
                else:
                    logger.info(
                        "[Retrieve] All sub-queries had local results — "
                        "web augmentation using full question as fallback."
                    )

            web_state = {**state, "question": web_query}
            web_result = web_search(web_state)
            web_docs = web_result.get("documents", [])
            added = 0
            for doc in web_docs:
                key = doc.page_content[:200]
                if key not in seen_keys:
                    # Tag web docs so the LLM prompt and source list distinguish them
                    doc.metadata.setdefault("retrieval_method", "web_comparison")
                    all_docs.append(doc)
                    seen_keys.add(key)
                    added += 1
            if added:
                logger.info(f"[Retrieve] Added {added} web doc(s) for comparison context.")
                if "Web Comparison Augmentation" not in techniques:
                    techniques.append("Web Comparison Augmentation")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[Retrieve] Comparison web augmentation failed (non-fatal): {exc}")

    return {"documents": all_docs, "techniques_used": techniques}


def _node_web_search(state: GraphState) -> GraphState:
    return web_search(state)


def _node_query_rewrite(state: GraphState) -> GraphState:
    """Rewrite then immediately re-retrieve."""
    new_state = rewrite_query(state)
    # Merge and re-retrieve with the rewritten query.
    merged = {**state, **new_state}
    # Bypass HyDE for rewritten queries (they are already optimized search queries)
    retrieved = retrieve(merged, get_hybrid_retriever(), get_hyde_retriever(), use_hyde=False)
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
    """Pull similar past examples and stash the few-shot prefix in state (thread-safe)."""
    try:
        prefix = get_few_shot_learner().get_dynamic_prompt(state["question"])
    except Exception:
        prefix = ""
    techniques = list(state.get("techniques_used", []))
    if prefix and "Dynamic Few-Shot" not in techniques:
        techniques.append("Dynamic Few-Shot")
    return {"few_shot_prefix": prefix, "techniques_used": techniques}


def _node_generate(state: GraphState) -> GraphState:
    prefix = state.get("few_shot_prefix", "")
    return generate(state, few_shot_prefix=prefix)


def _node_regenerate(state: GraphState) -> GraphState:
    """Re-generate after hallucination failure. Bumps regen_count."""
    regen_count = state.get("regen_count", 0) + 1
    logger.info(f"---REGENERATE #{regen_count}---")
    prefix = state.get("few_shot_prefix", "")
    new_state = {**state, "regen_count": regen_count}
    gen = generate(new_state, few_shot_prefix=prefix)
    gen["regen_count"] = regen_count
    return gen


def _node_direct_llm(state: GraphState) -> GraphState:
    """Tag the route and pass through — GENERATE node handles actual generation.

    Previously this called generate() internally, which caused a double-generation:
    _node_direct_llm ran generate() AND then GENERATE node ran it again, wasting
    one full LLM call on every conversational/greeting/math query.

    Now: just ensure documents is empty so GENERATE skips retrieval context.
    The _after_generate conditional edge already routes direct_llm → finalize,
    bypassing hallucination grading correctly.
    """
    logger.info("---DIRECT LLM (no retrieval) — handing off to GENERATE node---")
    return {"documents": [], "route": "direct_llm"}


def _node_end_with_sources(state: GraphState) -> GraphState:
    """Terminal node: build the sources list from documents."""
    if state.get("clarification_needed", False):
        return {"sources": []}

    import re
    sources_dict = {}
    for d in state.get("documents", []):
        raw_source = d.metadata.get("source", "unknown")
        # Clean the uuid hash prefix (e.g., c1d5e99e_filename.pdf -> filename.pdf)
        clean_source = re.sub(r'^[a-fA-F0-9]{8}_', '', raw_source)
        
        page = d.metadata.get("page_number")
        if page in (None, "", "?", "unknown"):
            page_val = None
        else:
            try:
                page_val = int(page)
            except ValueError:
                page_val = str(page).strip()
                
        doc_type = d.metadata.get("doc_type", "")
        excerpt = d.page_content[:150].replace("\n", " ") + "..."
        
        if clean_source not in sources_dict:
            sources_dict[clean_source] = {
                "pages": set(),
                "doc_type": doc_type,
                "excerpts": []
            }
        
        if page_val is not None:
            sources_dict[clean_source]["pages"].add(page_val)
        if excerpt:
            sources_dict[clean_source]["excerpts"].append(excerpt)

    sources = []
    for clean_source, info in sources_dict.items():
        # Sort pages: ints first, then strings
        pages_list = sorted(list(info["pages"]), key=lambda x: (isinstance(x, str), x))
        pages_str = ", ".join(map(str, pages_list)) if pages_list else None
        
        combined_excerpt = " | ".join(info["excerpts"][:3])
        
        sources.append({
            "source": clean_source,
            "page": pages_str,
            "doc_type": info["doc_type"],
            "excerpt": combined_excerpt,
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

    # grade_hallucination → {regenerate | grade_answer}
    workflow.add_conditional_edges(
        GRADE_HALLUCINATION,
        decide_after_hallucination,
        {GO_REGENERATE: REGENERATE, GO_CONFIDENCE: GRADE_ANSWER},
    )

    # regenerate → grade_hallucination (loop)
    workflow.add_edge(REGENERATE, GRADE_HALLUCINATION)

    # grade_answer → confidence_scorer (or web_search if answer doesn't address the question)
    # This was previously a straight edge, meaning decide_after_answer was never called.
    # Fixed: now uses the conditional edge so web-search fallback actually fires.
    def _after_answer(state: GraphState) -> str:
        from config import SelfCorrectionConfig
        if not state.get("answer_addresses_question", False):
            if state.get("retry_count", 0) < SelfCorrectionConfig.MAX_RETRY_COUNT:
                logger.info("[GradeAnswer] Answer does not address question — routing to web search.")
                return WEBSEARCH
        return CONFIDENCE_SCORER

    workflow.add_conditional_edges(
        GRADE_ANSWER,
        _after_answer,
        {WEBSEARCH: WEBSEARCH, CONFIDENCE_SCORER: CONFIDENCE_SCORER},
    )

    # confidence_scorer → {finalize | web_search}
    def _after_confidence(state: GraphState) -> str:
        # If the answer does not address the question and we haven't hit the retry limit, retry retrieval
        if not state.get("answer_addresses_question", False):
            if state.get("retry_count", 0) < SelfCorrectionConfig.MAX_RETRY_COUNT:
                logger.info("Answer does not address the question. Routing to WEBSEARCH for retry.")
                return WEBSEARCH
        return "finalize"

    workflow.add_conditional_edges(
        CONFIDENCE_SCORER,
        _after_confidence,
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
def run_query(
    question: str,
    config: dict | None = None,
    owner_id: str = "default_owner",
    session_id: str = ""
) -> dict:
    """Run a single question through the full pipeline. Returns the final state."""
    from llm import get_request_audit, start_request_context

    start = time.time()
    start_request_context()

    state = initial_state(question, owner_id=owner_id, session_id=session_id)
    app = get_app()
    logger.info(f"=== RUN QUERY: {question} owner={owner_id} session={session_id} ===")
    try:
        final_state = app.invoke(state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Graph execution failed: {exc}")
        final_state = {**state, "generation": f"[Pipeline error: {exc}]"}

    elapsed = time.time() - start
    final_state["processing_time"] = round(elapsed, 3)

    audit = get_request_audit()
    final_state["llm_calls"] = audit.get("llm_calls", 0)
    final_state["skipped_nodes"] = audit.get("skipped_nodes", [])
    final_state["fallback_nodes"] = audit.get("fallbacks", [])
    telemetry = audit.get("telemetry", [])
    final_state["node_telemetry"] = telemetry

    skipped_str = ", ".join(final_state["skipped_nodes"]) if final_state["skipped_nodes"] else "None"
    fallbacks_str = ", ".join(final_state["fallback_nodes"]) if final_state["fallback_nodes"] else "None"

    # Pipeline Trace formatting
    trace_lines = []
    slowest_node = "N/A"
    slowest_dur = -1
    total_llm_dur_ms = 0

    for item in telemetry:
        node_name = item.get("node", "Unknown")
        dur = item.get("duration_ms", 0)
        provider = item.get("provider", "local")
        cached = " (CACHE HIT)" if item.get("cached") else ""
        timeout_flag = " (TIMEOUT)" if item.get("timeout") else ""
        fallback_flag = " (FALLBACK)" if item.get("fallback") else ""

        if dur > slowest_dur:
            slowest_dur = dur
            slowest_node = f"{node_name} ({dur / 1000.0:.2f} s)"

        total_llm_dur_ms += dur
        trace_lines.append(f"  {node_name:<22} {dur:>6} ms  [{provider}]{cached}{timeout_flag}{fallback_flag}")

    avg_llm_latency_s = (total_llm_dur_ms / len(telemetry) / 1000.0) if telemetry else 0.0
    trace_block = "\n".join(trace_lines) if trace_lines else "  (No LLM calls invoked)"

    logger.info(
        f"\n------------------------------------------------------------------------\n"
        f"Pipeline Trace\n"
        f"{trace_block}\n"
        f"------------------------------------------------------------------------\n"
        f"Execution Summary\n"
        f"Total Graph Time:    {elapsed:.2f} s\n"
        f"Total LLM Calls:     {final_state['llm_calls']}\n"
        f"Average LLM Latency: {avg_llm_latency_s:.2f} s\n"
        f"Slowest Node:        {slowest_node}\n"
        f"Skipped Nodes:       {skipped_str}\n"
        f"Fallbacks:           {fallbacks_str}\n"
        f"Confidence:          {final_state.get('confidence_score', 0):.2f}\n"
        f"------------------------------------------------------------------------"
    )
    return final_state


def stream_query(
    question: str,
    config: dict | None = None,
    owner_id: str = "default_owner",
    session_id: str = ""
):
    """Stream intermediate states (for the SSE pipeline-trace endpoint).
    
    The final __done__ event now contains the complete final_state so the
    frontend does NOT need to issue a second POST /api/query to get the answer.
    """
    import time as _time
    start = _time.time()
    state = initial_state(question, owner_id=owner_id, session_id=session_id)
    app = get_app()
    final_state = state  # will be overwritten by the last node update
    all_updates: dict = {}

    for chunk in app.stream(state, config=config, stream_mode="updates"):
        # Each chunk is {node_name: state_update}
        for node_name, update in chunk.items():
            if update is None:
                update = {}
            elapsed = round(_time.time() - start, 3)
            yield {"node": node_name, "update": _safe_update(update), "elapsed": elapsed}
            # Accumulate updates so we can reconstruct the final state
            if isinstance(update, dict):
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


def _safe_update(update: dict | None) -> dict:
    """Make a state update JSON-serializable for streaming."""
    if not update or not hasattr(update, "items"):
        return {}
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
