"""
GraphState — the single TypedDict that flows through every LangGraph node.

This state carries the question, all intermediate retrieval/self-correction
results, the final generation, and every quality signal used to compute the
final confidence score.
"""

from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # ---- Input ----
    question: str

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

    # ---- Generation ----
    generation: str                     # final answer

    # ---- Quality gates ----
    hallucination_free: bool            # hallucination grader result
    hallucination_score: float          # 0.0-1.0 grounding score
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


def initial_state(question: str) -> GraphState:
    """Construct a blank state for a new query."""
    return GraphState(
        question=question,
        sub_questions=[],
        query_variations=[],
        documents=[],
        parent_documents=[],
        crag_state="",
        contradiction_found=False,
        contradiction_detail="",
        clarification_needed=False,
        clarification_question="",
        generation="",
        hallucination_free=False,
        hallucination_score=0.0,
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
    )
