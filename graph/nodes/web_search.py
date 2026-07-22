"""
Web search node (Tavily).

Triggered when:
  - crag_state == "incorrect" (docs irrelevant → augment from web), or
  - retry exhausted (insufficient docs after re-queries), or
  - the router picked "websearch" directly, or
  - the answer grader marked a generation "not useful".

Uses langchain-tavily. Appends web results as Documents to the state and flags
web_search_used = True.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

from config import APIKeys, SelfCorrectionConfig
from graph.state import GraphState

WEBSEARCH = "web_search"


class WebSearchUnavailable(RuntimeError):
    pass


def _get_tavily_tool():
    """Lazily build the Tavily search tool. Raises if no API key."""
    if not APIKeys.TAVILY_API_KEY:
        raise WebSearchUnavailable(
            "TAVILY_API_KEY is not set. Web search is unavailable. "
            "Add it to .env to enable web fallback."
        )
    from langchain_tavily import TavilySearch

    return TavilySearch(
        max_results=SelfCorrectionConfig.WEB_SEARCH_MAX_RESULTS,
        tavily_api_key=APIKeys.TAVILY_API_KEY,
    )


_tavily_tool = None


def get_tavily_tool():
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = _get_tavily_tool()
    return _tavily_tool


def web_search(state: GraphState, query_override: Optional[str] = None) -> GraphState:
    """Run a Tavily web search and append results to documents."""
    question = query_override or state["question"]
    documents: List[Document] = list(state.get("documents", []))
    logger.info(f"---WEB SEARCH--- for: {question[:60]}")

    try:
        tool = get_tavily_tool()
        response = tool.invoke({"query": question})
        # Tavily returns either a list of results or {"results": [...]}.
        results = response.get("results", response) if isinstance(response, dict) else response
        added = 0
        for r in results:
            content = (r.get("content") or "").strip()
            url = r.get("url", "")
            title = r.get("title", "")
            if content:
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": url or title or "web",
                        "source_path": url,
                        "doc_type": "web",
                        "page_number": 1,
                        "ocr_confidence": 1.0,
                        "retrieval_method": "web_search",
                    },
                ))
                added += 1
        logger.info(f"---WEB SEARCH: added {added} web documents---")
    except WebSearchUnavailable as exc:
        logger.warning(f"Web search skipped: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Web search failed: {exc}")

    techniques = list(state.get("techniques_used", []))
    if "Web Search (Tavily)" not in techniques:
        techniques.append("Web Search (Tavily)")

    return {
        "documents": documents,
        "web_search_used": True,
        "techniques_used": techniques,
    }
