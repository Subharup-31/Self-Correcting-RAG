"""
Answer grader node.

Assesses whether the final answer actually addresses / resolves the user's
question. Ported from Agentic-Adaptive-RAG (with the {question}/{generation}
template bug fixed).

Sets answer_addresses_question = bool.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from graph.state import GraphState
from llm import get_grader_llm

GRADE_ANSWER = "grade_answer"

# Next-step constants
GO_END_USEFUL = "end_useful"
GO_WEB_NOT_USEFUL = "web_search_not_useful"


class AnswerGrade(BaseModel):
    """Binary grade: does the answer resolve the question?"""
    addresses: bool = Field(
        description="True if the answer addresses/resolves the user's question"
    )
    reason: str = Field(description="One-line justification")


ANSWER_SYSTEM = """You are a grader assessing whether an answer addresses / \
resolves a user question.

Give a binary grade:
- True: the answer directly and substantively addresses the question.
- False: the answer is off-topic, evasive, incomplete, or says it cannot answer.

Be strict — an answer that merely mentions the topic without resolving the \
specific question is False."""

ANSWER_HUMAN = """User question:
{question}

Generated answer:
{generation}

Does the answer address the question?"""


def build_answer_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM), ("human", ANSWER_HUMAN)]
    )
    return prompt | llm.with_structured_output(AnswerGrade)


_answer_chain = None


def get_answer_chain():
    global _answer_chain
    if _answer_chain is None:
        _answer_chain = build_answer_chain()
    return _answer_chain


def grade_answer(state: GraphState) -> GraphState:
    """Grade whether the answer addresses the question."""
    question = state["question"]
    generation = state.get("generation", "")
    logger.info(f"---GRADE ANSWER--- for '{question[:50]}'")

    try:
        result: AnswerGrade = get_answer_chain().invoke(
            {"question": question, "generation": generation}
        )
        addresses = bool(result.addresses)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Answer grader failed: {exc}")
        addresses = True  # fail open

    logger.info(f"---ANSWER ADDRESSES QUESTION: {addresses}---")
    return {"answer_addresses_question": addresses}


def decide_after_answer(state: GraphState) -> str:
    """Conditional edge: useful → END; not useful → web search fallback."""
    if state.get("answer_addresses_question", False):
        return GO_END_USEFUL
    # Only fall back to web if we haven't exhausted retries.
    from config import SelfCorrectionConfig
    if state.get("retry_count", 0) < SelfCorrectionConfig.MAX_RETRY_COUNT:
        return GO_WEB_NOT_USEFUL
    return GO_END_USEFUL
