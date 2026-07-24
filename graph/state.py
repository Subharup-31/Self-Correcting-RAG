"""
GraphState — the single TypedDict that flows through every LangGraph node.

This state carries the question, all intermediate retrieval/self-correction
results, the final generation, and every quality signal used to compute the
final confidence score.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # ---- Input ----
    question: str
    owner_id: str
    session_id: str
    few_shot_prefix: str                # dynamic few-shot prompt context (thread-safe)


    # ---- Query processing ----
    sub_questions: List[str]            # from query decomposer
    query_variations: List[str]         # from multi-query expansion

    # ---- Retrieval ----
    documents: List[Document]           # retrieved + filtered child chunks
    parent_documents: List[Document]    # expanded parent chunks for context

    # ---- Self-correction states ----
    crag_state: str                     # "correct" | "ambiguous" | "incorrect"
    contradiction_found: bool           # docs contradict each other
    contradiction_detail: str           # "Doc 1 says X but Doc 2 says Y"

    # ---- Clarification ----
    clarification_needed: bool          # trigger clarifying question
    clarification_question: str         # the clarifying question text
    clarification_options: List[str]    # multiple-choice options for clarification

    # ---- Generation ----
    generation: str                     # final answer

    # ---- Quality gates ----
    hallucination_free: Optional[bool]  # True/False or None if grading unavailable
    hallucination_score: Optional[float] # 0.0-1.0 or None if grading unavailable
    unsupported_claims: List[str]       # specific claims not in docs
    answer_addresses_question: bool     # answer grader result

    # ---- Confidence ----
    confidence_score: float             # 0.0 - 1.0
    low_confidence: bool                # True if < CONFIDENCE_THRESHOLD
    confidence_reason: str              # why confidence is low

    # ---- Control flow ----
    route: str                          # "vectorstore" | "websearch" | "direct_llm"
    web_search_used: bool
    retry_count: int                    # re-query iteration counter (max 3)
    regen_count: int                    # regeneration counter (max 3)

    # ---- Audit ----
    techniques_used: List[str]          # which techniques fired
    sources: List[dict]                 # source citations
    processing_time: float
    skipped_nodes: List[str]            # nodes skipped during run
    fallback_nodes: List[str]           # nodes that used fallback execution
    llm_calls: int                      # total LLM calls invoked in request
    node_telemetry: List[dict]          # granular per-node timing and provider metrics


def initial_state(question: str, owner_id: str = "default_owner", session_id: str = "") -> GraphState:
    """Construct a blank state for a new query."""
    return GraphState(
        question=question,
        owner_id=owner_id,
        session_id=session_id,
        few_shot_prefix="",
        sub_questions=[],
        query_variations=[],
        documents=[],
        parent_documents=[],
        crag_state="",
        contradiction_found=False,
        contradiction_detail="",
        clarification_needed=False,
        clarification_question="",
        clarification_options=[],
        generation="",
        hallucination_free=None,
        hallucination_score=None,
        unsupported_claims=[],
        answer_addresses_question=False,
        confidence_score=0.0,
        low_confidence=False,
        confidence_reason="",
        route="",
        web_search_used=False,
        retry_count=0,
        regen_count=0,
        techniques_used=[],
        sources=[],
        processing_time=0.0,
        skipped_nodes=[],
        fallback_nodes=[],
        llm_calls=0,
        node_telemetry=[],
    )
