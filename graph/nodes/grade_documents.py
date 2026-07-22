"""
3-state CRAG document grader — the core self-correction mechanism.

For EACH retrieved document, classify as:
  - "correct"   → document is clearly relevant, use it
  - "ambiguous" → document partially overlaps but is unclear/vague
  - "incorrect" → document is irrelevant, discard it

Aggregation logic → sets state["crag_state"]:
  - All correct                                   → "correct"    → proceed
  - Any ambiguous + no incorrect                  → "ambiguous"  → clarify
  - Any incorrect (majority), or all filtered out → "incorrect"  → web search
"""

from __future__ import annotations

from typing import List, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from graph.state import GraphState
from llm import get_grader_llm

GRADE_DOCUMENTS = "grade_documents"

# Next-step constants (returned by the conditional edge)
GO_DETECT_CONTRADICTION = "detect_contradiction"
GO_CLARIFY = "clarify"
GO_WEB_SEARCH = "web_search_after_incorrect"


class DocumentGrade(BaseModel):
    """3-state relevance grade for a single retrieved document."""
    grade: Literal["correct", "ambiguous", "incorrect"]
    reason: str = Field(description="Short justification for the grade")


GRADER_SYSTEM = """You are a strict relevance grader assessing whether a retrieved \
document is useful for answering a user's question.

Grade the document as one of:
- "correct": the document clearly and directly contains information that helps \
answer the question.
- "ambiguous": the document is topically related but vague, partial, or unclear \
about the specific question.
- "incorrect": the document is irrelevant or off-topic for the question.

Be strict. Only mark "correct" when the document genuinely helps answer the \
specific question asked."""

GRADER_HUMAN = """Question: {question}

Retrieved document:
{document}

Grade this document's relevance (correct / ambiguous / incorrect) and explain why."""


def build_grader_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", GRADER_SYSTEM), ("human", GRADER_HUMAN)]
    )
    return prompt | llm.with_structured_output(DocumentGrade)


_grader_chain = None


def get_grader_chain():
    global _grader_chain
    if _grader_chain is None:
        _grader_chain = build_grader_chain()
    return _grader_chain


def grade_documents(state: GraphState) -> GraphState:
    """Grade each retrieved document and aggregate into a crag_state."""
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    logger.info(f"---GRADE DOCUMENTS--- {len(documents)} doc(s)")

    kept: List[Document] = []
    grades: dict[str, int] = {"correct": 0, "ambiguous": 0, "incorrect": 0}
    chain = get_grader_chain()

    for doc in documents:
        # Truncate very long documents to keep the LLM call cheap.
        snippet = doc.page_content[:1500]
        try:
            result: DocumentGrade = chain.invoke(
                {"question": question, "document": snippet}
            )
            grade = result.grade
            reason = result.reason
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Grader failed for a doc, keeping it: {exc}")
            grade, reason = "ambiguous", "grader error fallback"

        doc.metadata["grade"] = grade
        doc.metadata["grade_reason"] = reason
        grades[grade] = grades.get(grade, 0) + 1

        if grade in ("correct", "ambiguous"):
            kept.append(doc)

    # Aggregation → CRAG state.
    if not documents or not kept:
        crag_state = "incorrect"
    elif grades["incorrect"] == 0 and grades["ambiguous"] > 0:
        crag_state = "ambiguous"
    elif grades["incorrect"] > 0 and grades["correct"] == 0:
        crag_state = "incorrect"
    elif grades["incorrect"] >= grades["correct"]:
        # Majority irrelevant.
        crag_state = "incorrect"
    else:
        crag_state = "correct"

    techniques = list(state.get("techniques_used", []))
    if "CRAG (3-state grading)" not in techniques:
        techniques.append("CRAG (3-state grading)")

    logger.info(
        f"---GRADED: correct={grades['correct']} ambiguous={grades['ambiguous']} "
        f"incorrect={grades['incorrect']} → crag_state={crag_state}"
    )
    return {
        "documents": kept,
        "crag_state": crag_state,
        "techniques_used": techniques,
    }


def decide_after_grading(state: GraphState) -> str:
    """Conditional-edge mapper from grade_documents."""
    crag = state.get("crag_state", "correct")
    if crag == "correct":
        return GO_DETECT_CONTRADICTION
    if crag == "ambiguous":
        return GO_CLARIFY
    return GO_WEB_SEARCH
