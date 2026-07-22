"""
Clarification node.

Triggered when crag_state == "ambiguous". Generates one targeted clarifying
question that would help disambiguate the user's intent and narrow retrieval.

Sets clarification_needed = True and clarification_question = "...".
The API will return this to the user INSTEAD of a hallucinated answer.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from graph.state import GraphState
from llm import get_grader_llm

CLARIFY = "clarify"


class ClarifyingQuestion(BaseModel):
    """A single clarifying question to ask the user."""
    question: str = Field(
        description="A concise, specific clarifying question that helps narrow the answer"
    )


CLARIFY_SYSTEM = """The user asked a question, but the available documents are \
ambiguous or only partially relevant. Rather than guess, generate ONE short, \
specific clarifying question that would help disambiguate what the user wants.

Good clarifying questions:
- Narrow scope ("Are you asking about X in context Y or Z?")
- Identify the specific entity/aspect of interest
- Are answerable in a few words by the user
- Reference the ambiguity you observed

Do NOT answer the original question. Do NOT apologize. Just ask the clarifying \
question directly."""

CLARIFY_HUMAN = """Original question: {question}

Ambiguous retrieved context (snippet):
{context}

Generate one clarifying question:"""


def build_clarify_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", CLARIFY_SYSTEM), ("human", CLARIFY_HUMAN)]
    )
    return prompt | llm.with_structured_output(ClarifyingQuestion)


_clarify_chain = None


def get_clarify_chain():
    global _clarify_chain
    if _clarify_chain is None:
        _clarify_chain = build_clarify_chain()
    return _clarify_chain


def clarify(state: GraphState) -> GraphState:
    """Generate a clarifying question and flag the state for clarification."""
    question = state["question"]
    documents = state.get("documents", [])
    context = "\n".join(d.page_content[:400] for d in documents[:3]) or "(no docs)"
    logger.info(f"---CLARIFY--- for question: {question[:50]}")

    try:
        result: ClarifyingQuestion = get_clarify_chain().invoke(
            {"question": question, "context": context}
        )
        clarifying_q = result.question.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Clarify chain failed: {exc}")
        clarifying_q = (
            "Could you clarify what specific aspect you're asking about? "
            "The available documents cover several related topics."
        )

    techniques = list(state.get("techniques_used", []))
    if "Clarifying Question" not in techniques:
        techniques.append("Clarifying Question")

    logger.info(f"---CLARIFY QUESTION: {clarifying_q}---")
    return {
        "clarification_needed": True,
        "clarification_question": clarifying_q,
        "techniques_used": techniques,
    }
