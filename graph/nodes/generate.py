"""
Generation node.

Produces the final answer from the retrieved (and graded) documents. Optionally:
  - Uses parent-document expansion for richer context
  - Injects few-shot examples from the learning manager
  - Surfaces contradictions explicitly instead of ignoring them
  - Cites sources
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from graph.state import GraphState
from llm import get_generation_llm

GENERATE = "generate"


GENERATION_SYSTEM = """You are a precise, trustworthy question-answering assistant. \
You answer ONLY using the provided context documents. Follow these rules strictly:

1. If the context contains the answer, answer concisely and cite the source \
document name and page in square brackets, e.g. [company_report.pdf, p.3].
2. If the context is insufficient to fully answer, say so explicitly: \
"The provided documents do not contain enough information to fully answer this." \
Do NOT invent or guess missing facts.
3. If you are aware the documents CONTRADICT each other on a point, say so \
explicitly: "Note: the sources disagree — [Doc A] says X while [Doc B] says Y." \
Do not silently pick one side.
4. Keep the answer focused and factual. No filler, no preamble."""

GENERATION_HUMAN = """Context documents:
{context}

User question: {question}

Answer:"""


def _format_context(documents: List[Document], max_chars_per_doc: int = 1500) -> str:
    """Format retrieved documents into a labeled context block."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_number", "?")
        text = doc.page_content[:max_chars_per_doc].strip()
        blocks.append(f"[Doc {i} — {source}, p.{page}]\n{text}")
    return "\n\n".join(blocks) if blocks else "(no context documents)"


def build_generation_chain(few_shot_prefix: str = ""):
    """Build the generation chain. few_shot_prefix is optional instructional text."""
    llm = get_generation_llm()
    system = GENERATION_SYSTEM
    if few_shot_prefix:
        system = few_shot_prefix + "\n\n" + system
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", GENERATION_HUMAN)]
    )
    return prompt | llm | StrOutputParser()


def generate(state: GraphState, few_shot_prefix: str = "") -> GraphState:
    """Generate an answer grounded in the retrieved documents."""
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    logger.info(f"---GENERATE--- {len(documents)} doc(s) for '{question[:50]}'")

    context = _format_context(documents)
    chain = build_generation_chain(few_shot_prefix=few_shot_prefix)

    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Generation failed: {exc}")
        answer = (
            "I was unable to generate an answer due to an internal error. "
            "Please try again."
        )

    techniques = list(state.get("techniques_used", []))
    if "Generation" not in techniques:
        techniques.append("Generation")

    logger.info(f"---GENERATED ({len(answer)} chars)---")
    return {"generation": answer, "techniques_used": techniques}
