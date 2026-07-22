"""
3-way query router using Gemini with structured output.

Routes to:
  - "vectorstore": question is domain-specific and docs are available
  - "websearch"  : real-time / current-events info needed
  - "direct_llm" : conversational, no retrieval needed (greetings, math, chit-chat)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from graph.state import GraphState
from llm import get_grader_llm

# Node names (used as graph node ids + return values from conditional edges)
ROUTE = "route_question"
RETRIEVE = "retrieve"
WEBSEARCH = "web_search"
DIRECT_LLM = "direct_llm"


class RouteQuery(BaseModel):
    """Route a user query to the most appropriate data source."""
    datasource: Literal["vectorstore", "websearch", "direct_llm"] = Field(
        description="Route the question to the most appropriate source"
    )
    reason: str = Field(description="One-line explanation for the routing decision")


ROUTER_SYSTEM = """You are an expert at routing a user question to one of three sources.

Choose:
- "vectorstore": the question is about specific facts, concepts, definitions, \
comparisons, or information that would be found in a document collection. \
Topics like AI, RAG, retrieval, embeddings, finance reports, technical manuals, \
company information, research notes, Nexora Technologies, hackathon details, \
or ANY question referencing "my", "the document", "uploaded file", "this presentation", "my idea".
- "websearch": the question needs real-time, current, or frequently-updating \
information — news, stock prices, today's weather, recent events, live data (unless it is about the user's uploaded project/idea, in which case default to vectorstore).
- "direct_llm": the question is conversational, a greeting, simple math, \
opinion, or general chit-chat that does not require document retrieval.

Default to "vectorstore" when in doubt about factual/conceptual questions. Always default to "vectorstore" if the user mentions "my idea" or "my project" or "my document"."""

ROUTER_HUMAN = "Question to route: {question}"


def build_router_chain():
    """Build the structured-output router chain."""
    llm = get_grader_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_SYSTEM), ("human", ROUTER_HUMAN)]
    )
    return prompt | llm.with_structured_output(RouteQuery)


_router_chain = None


def get_router_chain():
    global _router_chain
    if _router_chain is None:
        _router_chain = build_router_chain()
    return _router_chain


def route_question(state: GraphState) -> GraphState:
    """Classify the question and record the chosen route in state."""
    question = state["question"]
    logger.info(f"---ROUTE QUESTION--- {question[:60]}")
    try:
        result: RouteQuery = get_router_chain().invoke({"question": question})
        route = result.datasource
        reason = result.reason
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Router failed, defaulting to vectorstore: {exc}")
        route, reason = "vectorstore", "router fallback"

    logger.info(f"---ROUTE: {route} ({reason})---")
    techniques = list(state.get("techniques_used", []))
    if "Routing" not in techniques:
        techniques.append("Routing")
    return {"route": route, "techniques_used": techniques}


def route_decision(state: GraphState) -> str:
    """Conditional-edge mapper: returns the next node name based on the route."""
    route = state.get("route", "vectorstore")
    if route == "websearch":
        return WEBSEARCH
    if route == "direct_llm":
        return DIRECT_LLM
    return RETRIEVE
