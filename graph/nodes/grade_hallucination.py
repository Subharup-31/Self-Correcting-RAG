"""
Hallucination grader.

Ported from Agentic-Adaptive-RAG/graph/chains/hallucination_grader.py, but
UPGRADED from a binary yes/no to a richer grade:
  - grounded: bool                       (is the answer supported by the docs?)
  - confidence_contribution: float 0-1   (how well grounded)
  - unsupported_claims: List[str]        (specific claims not in the docs)

The numeric score feeds the confidence scorer; the unsupported_claims list can
be surfaced to the user for transparency.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from graph.state import GraphState
from llm import get_grader_llm

GRADE_HALLUCINATION = "grade_hallucination"

# Next-step constants
GO_REGENERATE = "regenerate"
GO_CONFIDENCE = "confidence"


class HallucinationGrade(BaseModel):
    """Rich hallucination assessment of a generated answer vs its sources."""
    grounded: bool = Field(
        description="True if the answer is fully supported by the provided documents"
    )
    confidence_contribution: float = Field(
        ge=0.0, le=1.0,
        description="How well the answer is grounded: 1.0 fully grounded, 0.0 entirely unsupported",
    )
    unsupported_claims: List[str] = Field(
        description="Specific factual claims in the answer that are NOT supported by the documents"
    )


HALLUCINATION_SYSTEM = """You are a strict hallucination grader. Given a set of \
factual documents and an LLM-generated answer, determine whether the answer is \
GROUND IN (supported by) the documents.

Assess:
- grounded: True only if every factual claim in the answer can be traced to the documents.
- confidence_contribution: a score in [0.0, 1.0] reflecting how well-grounded the \
answer is overall (1.0 = fully grounded, 0.0 = completely made up, 0.5 = mixed).
- unsupported_claims: list each specific factual claim in the answer that is NOT \
supported by the documents. Empty list if all claims are supported.

Be strict. Claims that paraphrase or logically follow from the documents are \
supported; claims that introduce new facts, numbers, names, or conclusions not \
present in the documents are unsupported."""

HALLUCINATION_HUMAN = """Documents:
{documents}

Generated answer:
{generation}"""


def build_hallucination_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", HALLUCINATION_SYSTEM), ("human", HALLUCINATION_HUMAN)]
    )
    return prompt | llm.with_structured_output(HallucinationGrade)


_hallucination_chain = None


def get_hallucination_chain():
    global _hallucination_chain
    if _hallucination_chain is None:
        _hallucination_chain = build_hallucination_chain()
    return _hallucination_chain


def grade_hallucination(state: GraphState) -> GraphState:
    """Grade whether the generated answer is grounded in the documents."""
    documents: List[Document] = state.get("documents", [])
    generation = state.get("generation", "")
    logger.info(f"---GRADE HALLUCINATION--- ({len(documents)} docs)")

    docs_text = "\n\n".join(d.page_content[:1200] for d in documents) or "(no docs)"
    chain = get_hallucination_chain()

    try:
        result: HallucinationGrade = chain.invoke(
            {"documents": docs_text, "generation": generation}
        )
        grounded = bool(result.grounded)
        score = float(result.confidence_contribution)
        unsupported = list(result.unsupported_claims or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Hallucination grader failed: {exc}")
        grounded, score, unsupported = True, 0.5, []

    logger.info(
        f"---HALLUCINATION: grounded={grounded} score={score:.2f} "
        f"unsupported={len(unsupported)}---"
    )
    return {
        "hallucination_free": grounded,
        "hallucination_score": score,
        "unsupported_claims": unsupported,
    }


def decide_after_hallucination(state: GraphState) -> str:
    """Conditional edge: regenerate if not grounded (under retry cap)."""
    regen_count = state.get("regen_count", 0)
    from config import SelfCorrectionConfig

    if not state.get("hallucination_free", True):
        if regen_count < SelfCorrectionConfig.MAX_RETRY_COUNT:
            return GO_REGENERATE
    return GO_CONFIDENCE
