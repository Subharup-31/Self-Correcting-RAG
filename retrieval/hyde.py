"""
HyDE (Hypothetical Document Embeddings).

Strategy (original Gao et al. 2022):
  1. Ask the LLM to generate a hypothetical answer to the query (as if the docs
     already existed).
  2. Use that hypothetical answer as the search query against the retriever —
     because answer-to-answer matching is more semantically precise than
     question-to-answer matching.
  3. Discard the hypothetical; return the *real* retrieved documents.

Ported conceptually from Self-Healing-RAG/backend/hyde.py (which used LlamaIndex
built-ins) but reimplemented self-contained for LangChain + ChromaDB + Gemini.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from llm import get_hyde_llm
from retrieval.hybrid_retriever import HybridRetriever

HYDE_SYSTEM = """You are an expert knowledge assistant. Given a user question, \
write a short (3-5 sentence) hypothetical answer document that would directly \
answer the question. Write it as if it were a real passage from a textbook or \
technical document — confident, specific, factual in tone. Do not add \
disclaimers like "I don't know". This passage will be used only to find \
similar real documents, not shown to the user."""

HYDE_HUMAN = """Question: {question}

Hypothetical answer document:"""


class HyDERetriever:
    """Generate a hypothetical answer, then retrieve real docs matching it."""

    def __init__(self, retriever: HybridRetriever, llm=None):
        self.retriever = retriever
        self.llm = llm or get_hyde_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", HYDE_SYSTEM), ("human", HYDE_HUMAN)]
        )

    def generate_hypothetical_document(self, query: str) -> str:
        """Ask the LLM to produce a hypothetical answer passage."""
        import time
        from config import LLMTimeoutConfig
        from llm import invoke_chain_safe, record_audit_event

        start_time = time.time()
        logger.info("[HyDE] START")
        chain = self.prompt | self.llm
        try:
            response, was_cached = invoke_chain_safe(
                chain,
                {"question": query},
                timeout_seconds=LLMTimeoutConfig.HYDE_TIMEOUT,
                node_name="HyDE"
            )
            content = response.content if hasattr(response, "content") else response
            if isinstance(content, list):
                text = "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in content)
            else:
                text = str(content)
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[HyDE] DONE ({elapsed_ms} ms)")
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[HyDE] FAILED ({exc}). Fallback: original query")
            record_audit_event("fallback", f"HyDE ({exc})")
            return query

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        owner_id: str = "default_owner",
        session_id: str = ""
    ) -> List[Document]:
        """HyDE pipeline: generate hypothetical doc → retrieve real docs.

        Falls back gracefully to plain retrieval if hypothetical generation fails.
        """
        hyde_doc = self.generate_hypothetical_document(query)
        # If HyDE produced nothing usable, fall back to the original query.
        search_query = hyde_doc if len(hyde_doc) > 20 else query
        results = self.retriever.retrieve(
            query,
            top_k=top_k,
            owner_id=owner_id,
            session_id=session_id,
            vector_query=search_query,
            bm25_query=query,
        )
        for d in results:
            d.metadata.setdefault("technique", "HyDE")
        logger.info(f"HyDE retrieve '{query[:40]}...': {len(results)} docs")
        return results
