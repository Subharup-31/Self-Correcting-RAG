"""
Contradiction detection node.

Checks whether the top retrieved documents CONTRADICT each other (pairwise).
If a contradiction is found:
  - Sets contradiction_found = True
  - Records a human-readable contradiction_detail ("Doc A states X; Doc B states Y")
  - The final answer will surface this explicitly instead of silently picking one.

This is the explicit anti-hallucination mechanism for conflicting sources.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from graph.state import GraphState
from llm import get_grader_llm

DETECT_CONTRADICTION = "detect_contradiction"


class ContradictionCheck(BaseModel):
    """Result of checking a pair of documents for contradiction."""
    has_contradiction: bool = Field(
        description="True if the two documents make conflicting factual claims"
    )
    detail: str = Field(
        description='If contradiction: "Document A states X while Document B states Y". Else empty.'
    )


CONTRADICTION_SYSTEM = """You are a careful fact-checker. You will be shown two \
document passages and a question. Determine whether the two passages make \
CONFLICTING factual claims that are relevant to the question.

Only report a contradiction when the two passages state mutually incompatible \
facts (e.g. different numbers, dates, opposing conclusions) about the same \
specific point. Mere differences in detail, wording, or scope are NOT \
contradictions.

If you find a contradiction, write a precise detail string of the form: \
"Document A states <X> while Document B states <Y>"."""

CONTRADICTION_HUMAN = """Question: {question}

Document A:
{doc_a}

Document B:
{doc_b}"""


def build_contradiction_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", CONTRADICTION_SYSTEM), ("human", CONTRADICTION_HUMAN)]
    )
    return prompt | llm.with_structured_output(ContradictionCheck)


_contradiction_chain = None


def get_contradiction_chain():
    global _contradiction_chain
    if _contradiction_chain is None:
        _contradiction_chain = build_contradiction_chain()
    return _contradiction_chain


def detect_contradiction(state: GraphState) -> GraphState:
    """Check the top-3 retrieved documents pairwise for contradictions."""
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    chain = get_contradiction_chain()

    # Only check top few to bound cost.
    candidates = documents[:3]
    logger.info(f"---DETECT CONTRADICTION--- checking {len(candidates)} doc(s)")

    contradiction_found = False
    detail = ""
    for i in range(len(candidates)):
        if contradiction_found:
            break
        for j in range(i + 1, len(candidates)):
            try:
                result: ContradictionCheck = chain.invoke({
                    "question": question,
                    "doc_a": candidates[i].page_content[:1000],
                    "doc_b": candidates[j].page_content[:1000],
                })
                if result.has_contradiction:
                    contradiction_found = True
                    src_i = candidates[i].metadata.get("source", f"Doc {i+1}")
                    src_j = candidates[j].metadata.get("source", f"Doc {j+1}")
                    detail = f"[{src_i}] vs [{src_j}]: {result.detail}"
                    logger.warning(f"---CONTRADICTION FOUND: {detail}---")
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Contradiction check failed for pair ({i},{j}): {exc}")

    techniques = list(state.get("techniques_used", []))
    if contradiction_found and "Contradiction Detection" not in techniques:
        techniques.append("Contradiction Detection")

    return {
        "contradiction_found": contradiction_found,
        "contradiction_detail": detail,
        "techniques_used": techniques,
    }
