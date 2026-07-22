"""
Query rewrite node (for the re-query loop).

When retrieval quality is poor (crag_state == "incorrect"), rewrite the query
to be more effective and retry retrieval. Increments retry_count.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from graph.state import GraphState
from llm import get_grader_llm

QUERY_REWRITE = "query_rewrite"


QUERY_REWRITE_SYSTEM = """You are an expert at rewriting search queries for \
better document retrieval. Given an original question and the reason retrieval \
failed, produce a SINGLE rewritten query that:
- Is more specific and unambiguous
- Uses different vocabulary / synonyms that might match document wording
- Removes conversational filler
- Stays focused on the same information need
- CRITICAL: If the original question contains words like "my", "this document", "uploaded file", or references a specific user context (like "my idea", "my project"), DO NOT generalize it into an academic explanation query (e.g. do not turn "my problem statement" into "how to write a thesis problem statement"). Keep the search query focused on extracting details from the user's specific project files.

Output ONLY the rewritten query, nothing else."""

QUERY_REWRITE_HUMAN = """Original question: {question}

Reason previous retrieval was insufficient: {reason}

Rewritten query:"""


def build_query_rewrite_chain():
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", QUERY_REWRITE_SYSTEM), ("human", QUERY_REWRITE_HUMAN)]
    )
    from langchain_core.output_parsers import StrOutputParser
    return prompt | llm | StrOutputParser()


_rewrite_chain = None


def get_rewrite_chain():
    global _rewrite_chain
    if _rewrite_chain is None:
        _rewrite_chain = build_query_rewrite_chain()
    return _rewrite_chain


def rewrite_query(state: GraphState) -> GraphState:
    """Rewrite the question for a better retry. Bumps retry_count."""
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    reason = state.get("crag_state", "incorrect")
    logger.info(f"---QUERY REWRITE--- retry #{retry_count + 1} for '{question[:50]}'")

    try:
        rewritten = get_rewrite_chain().invoke(
            {"question": question, "reason": f"previous crag_state={reason}"}
        ).strip()
        if not rewritten or len(rewritten) < 5:
            rewritten = question
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Query rewrite failed, keeping original: {exc}")
        rewritten = question

    techniques = list(state.get("techniques_used", []))
    if "Query Rewriting" not in techniques:
        techniques.append("Query Rewriting")

    return {
        "question": rewritten,
        "retry_count": retry_count + 1,
        "techniques_used": techniques,
    }
