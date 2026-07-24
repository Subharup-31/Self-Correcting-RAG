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


def grade_documents(state: GraphState) -> dict:
    """Grade each retrieved document and aggregate into a crag_state in parallel."""
    import concurrent.futures
    import time
    from config import LLMTimeoutConfig
    from llm import invoke_chain_safe, record_audit_event

    start_time = time.time()
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    logger.info(f"[DocumentGrader] START ({len(documents)} docs)")

    if not documents:
        logger.info("[DocumentGrader] DONE (0 ms) -> 0 docs (incorrect)")
        return {
            "documents": [],
            "crag_state": "incorrect",
            "techniques_used": state.get("techniques_used", []),
        }

    chain = get_grader_chain()

    # Parallel grading via ThreadPoolExecutor using safe chain invocation
    def _grade_one(doc: Document):
        snippet = doc.page_content[:1500]
        try:
            res, _ = invoke_chain_safe(
                chain,
                {"question": question, "document": snippet},
                timeout_seconds=LLMTimeoutConfig.DOC_GRADER_TIMEOUT,
                node_name="DocumentGrader"
            )
            return res
        except Exception as exc:
            return exc

    max_workers = min(len(documents), 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_grade_one, documents))

    kept: List[Document] = []
    grades: dict[str, int] = {"correct": 0, "ambiguous": 0, "incorrect": 0}

    for doc, result in zip(documents, results):
        if isinstance(result, Exception):
            logger.warning(f"[DocumentGrader] Doc grading failed ({result}). Fallback: keep as correct")
            record_audit_event("fallback", f"DocumentGrader doc ({result})")
            grade, reason = "correct", f"grader error fallback: {result}"
        else:
            grade = result.grade
            reason = result.reason

        doc.metadata["grade"] = grade
        doc.metadata["grade_reason"] = reason
        grades[grade] = grades.get(grade, 0) + 1

        if grade in ("correct", "ambiguous"):
            kept.append(doc)

    # If all doc gradings failed or were kept, aggregate into CRAG state cleanly.
    if not kept:
        crag_state = "incorrect"
    elif grades["correct"] > 0:
        crag_state = "correct"
    elif grades["ambiguous"] > 0:
        crag_state = "ambiguous"
    else:
        crag_state = "incorrect"

    techniques = list(state.get("techniques_used", []))
    if "CRAG (3-state grading)" not in techniques:
        techniques.append("CRAG (3-state grading)")

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"[DocumentGrader] DONE ({elapsed_ms} ms) -> correct={grades['correct']} "
        f"ambiguous={grades['ambiguous']} incorrect={grades['incorrect']} → crag_state={crag_state}"
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
