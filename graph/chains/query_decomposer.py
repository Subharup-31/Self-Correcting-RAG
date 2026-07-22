"""
Query decomposition + multi-query expansion.

Ported from Self-Healing-RAG/backend/query_decomposition.py and upgraded:
  - decompose_query(): break complex questions into atomic sub-questions
  - multi_query_expansion(): generate 3 phrasing variations for robust retrieval
  - is_complex_query(): cheap heuristic to skip decomposition for simple queries

Uses Gemini via structured output (Pydantic).
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from llm import get_grader_llm

# Substrings that hint a query is complex (multi-part / comparison).
_COMPLEX_HINTS = (
    " and ", " vs ", " versus ", " compare ", " comparison ",
    " difference between ", " both ", " relate", " how does ", " why ",
    " what are the ", " explain how ",
)


def is_complex_query(query: str) -> bool:
    """Heuristic: should we bother decomposing this query?"""
    q = f" {query.lower()} "
    if any(hint in q for hint in _COMPLEX_HINTS):
        return True
    # Multiple question marks or "and" joiners also hint complexity.
    if q.count("?") > 1:
        return True
    return False


# ---------------------------------------------------------------------- #
# Decomposition
# ---------------------------------------------------------------------- #
class SubQueries(BaseModel):
    """Atomic sub-questions decomposed from a complex query."""
    questions: List[str] = Field(
        description="List of atomic, self-contained sub-questions"
    )


DECOMPOSE_SYSTEM = """You are an expert researcher. Break down a complex user \
query into simple, atomic sub-queries that a search engine can answer \
independently.

Guidelines:
- Each sub-query must be self-contained and answerable on its own.
- Aim for 2-4 sub-queries for genuinely complex questions.
- Keep sub-queries concise and focused.
- For simple questions, return just the original query (as a single-element list).
- Do NOT add sub-queries that weren't implied by the original question."""

DECOMPOSE_HUMAN = "User query: {query}\n\nDecompose into atomic sub-queries:"


def build_decompose_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", DECOMPOSE_SYSTEM), ("human", DECOMPOSE_HUMAN)]
    )
    return prompt | llm.with_structured_output(SubQueries)


_decompose_chain = None


def decompose_query(query: str) -> List[str]:
    """Decompose a complex query into atomic sub-questions."""
    if not is_complex_query(query):
        logger.debug(f"Query not complex, skipping decomposition: {query[:50]}")
        return []
    try:
        result: SubQueries = build_decompose_chain().invoke({"query": query})
        subs = [q.strip() for q in result.questions if q.strip()]
        logger.info(f"Decomposed '{query[:50]}' → {len(subs)} sub-queries")
        return subs
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Decomposition failed: {exc}")
        return []


# ---------------------------------------------------------------------- #
# Multi-query expansion (paraphrases for retrieval robustness)
# ---------------------------------------------------------------------- #
class QueryVariations(BaseModel):
    """Different phrasings of the same information need."""
    variations: List[str] = Field(
        description="3 alternative phrasings of the same question"
    )


MULTIQUERY_SYSTEM = """You are an expert at generating alternative phrasings of \
a question to improve document retrieval. Given a user question, produce THREE \
alternative phrasings that:
- Express the same information need
- Use different vocabulary and sentence structure
- Might match different document wordings

Return exactly 3 variations (plus imply the original). Do NOT change the topic."""

MULTIQUERY_HUMAN = "Original question: {query}\n\nGenerate 3 alternative phrasings:"


def build_multiquery_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", MULTIQUERY_SYSTEM), ("human", MULTIQUERY_HUMAN)]
    )
    return prompt | llm.with_structured_output(QueryVariations)


def multi_query_expansion(query: str) -> List[str]:
    """Generate alternative phrasings for retrieval."""
    try:
        result: QueryVariations = build_multiquery_chain().invoke({"query": query})
        variations = [v.strip() for v in result.variations if v.strip()]
        logger.info(f"Multi-query expansion → {len(variations)} variations")
        return variations[:3]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Multi-query expansion failed: {exc}")
        return []
