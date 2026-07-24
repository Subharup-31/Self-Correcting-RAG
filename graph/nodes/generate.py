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
document name and page number inside square brackets exactly in this format: [filename.ext, pg - PageNumber] (e.g. [company_report.pdf, pg - 3]).
2. If the page number is unknown, missing, or empty, use "?" (e.g. [company_report.pdf, pg - ?]).
3. If the context is insufficient to fully answer, say so explicitly: \
"The provided documents do not contain enough information to fully answer this." \
Do NOT invent or guess missing facts.
4. If you are aware the documents CONTRADICT each other on a point, say so \
explicitly: "Note: the sources disagree — [file_A.pdf, pg - X] says Y while [file_B.pdf, pg - Z] says W." \
Do not silently pick one side.
5. Keep the answer focused and factual. No filler, no preamble."""

GENERATION_HUMAN = """Context documents:
{context}

User question: {question}

Answer:"""


def _format_context(documents: List[Document], max_chars_per_doc: int = 1500) -> str:
    """Format retrieved documents into a labeled context block."""
    import re
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        # Clean the uuid hash prefix (e.g. c1d5e99e_filename.pdf -> filename.pdf)
        clean_source = re.sub(r'^[a-fA-F0-9]{8}_', '', source)
        page = doc.metadata.get("page_number")
        if page in (None, "", "?", "unknown"):
            page = "?"
        text = doc.page_content[:max_chars_per_doc].strip()
        blocks.append(f"[Document: {clean_source}, Page: {page}]\n{text}")
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


def _expand_to_parents(
    documents: List[Document],
    owner_id: str = "default_owner",
    session_id: str = ""
) -> tuple[List[Document], bool]:
    """Resolve child documents to their full parent chunks using the SQLite parent store."""
    from ingestion.parent_store import get_parents
    parent_ids = [d.metadata.get("parent_id") for d in documents if d.metadata.get("parent_id")]
    if not parent_ids:
        return documents, False

    parents_list = get_parents(parent_ids, owner_id=owner_id, session_id=session_id)
    parents_by_id = {p.metadata.get("parent_id"): p for p in parents_list if p.metadata.get("parent_id")}

    expanded = []
    seen_pids = set()
    used_expansion = False

    for d in documents:
        pid = d.metadata.get("parent_id")
        if pid and pid in parents_by_id:
            if pid not in seen_pids:
                expanded.append(parents_by_id[pid])
                seen_pids.add(pid)
                used_expansion = True
        else:
            expanded.append(d)

    return expanded, used_expansion


def generate(state: GraphState, few_shot_prefix: str = "") -> GraphState:
    """Generate an answer grounded in the retrieved documents."""
    import time
    from config import LLMTimeoutConfig
    from llm import invoke_chain_safe, record_audit_event

    start_time = time.time()
    question = state["question"]
    documents: List[Document] = state.get("documents", [])
    owner_id = state.get("owner_id", "default_owner")
    session_id = state.get("session_id", "")
    logger.info(f"[Generator] START ({len(documents)} docs)")

    # Expand children chunks to parent chunks for richer context, scoped to session/user
    expanded_docs, used_expansion = _expand_to_parents(documents, owner_id=owner_id, session_id=session_id)

    context = _format_context(expanded_docs)

    # Inject contradiction warning if detected by the prior node
    if state.get("contradiction_found", False) and state.get("contradiction_detail"):
        contradiction_warning = (
            "\n\n[CRITICAL WARNING: CONTRADICTING EVIDENCE DETECTED]\n"
            f"{state.get('contradiction_detail')}\n"
            "Please reconcile this conflict in your answer according to rule 3."
        )
        context += contradiction_warning
        logger.info("[Generator] Injected contradiction details into context.")

    chain = build_generation_chain(few_shot_prefix=few_shot_prefix)

    generation_failed = False
    try:
        answer, was_cached = invoke_chain_safe(
            chain,
            {"context": context, "question": question},
            timeout_seconds=LLMTimeoutConfig.GENERATION_TIMEOUT,
            node_name="Generator"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[Generator] Primary generation attempt failed ({exc}). Retrying with trimmed context...")
        try:
            # Fallback retry with top 3 docs context to reduce token payload
            trimmed_context = _format_context(expanded_docs[:3])
            answer, _ = invoke_chain_safe(
                chain,
                {"context": trimmed_context, "question": question},
                timeout_seconds=LLMTimeoutConfig.GENERATION_TIMEOUT,
                node_name="Generator-Fallback"
            )
            logger.info("[Generator] Retry succeeded with trimmed context.")
        except Exception as retry_exc:  # noqa: BLE001
            logger.error(f"[Generator] FAILED - generation unrecoverable ({retry_exc})")
            record_audit_event("fallback", f"Generator ({retry_exc})")
            generation_failed = True
            answer = "The system encountered a timeout while generating the answer. Please try re-submitting your query."

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info(f"[Generator] DONE ({elapsed_ms} ms) -> answer length {len(answer)} chars")

    techniques = list(state.get("techniques_used", []))
    if "Generation" not in techniques:
        techniques.append("Generation")
    if used_expansion and "Parent-Child Expansion" not in techniques:
        techniques.append("Parent-Child Expansion")

    res = {"generation": answer, "techniques_used": techniques, "generation_failed": generation_failed}
    if generation_failed:
        res["confidence_score"] = 0.0
        res["low_confidence"] = True
        res["confidence_reason"] = "Answer generation failed"
    return res
