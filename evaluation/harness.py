"""
Evaluation harness.

Runs all 12 test questions TWICE:
  1. Baseline: plain RAG with NO self-correction (vector search → generate, done)
  2. Ultimate: full self-correcting pipeline (HyDE, CRAG, contradiction detection,
     clarification, hallucination grading, confidence scoring, web fallback)

Metrics computed per run:
  - hallucination_rate   : % answers not grounded in docs
  - precision_rate       : % answers that address the question
  - self_correction_rate : % cases where the correction mechanism fired AND was correct
  - avg_confidence       : mean confidence score
  - clarification_rate   : % questions that triggered a clarifying question
  - low_confidence_rate  : % answers flagged as low confidence
  - web_search_rate      : % questions that used web search
  - contradiction_rate   : % questions that correctly flagged a contradiction

Outputs a comparison table to stdout and saves results to evaluation/results.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from config import EVALUATION_RESULTS
from evaluation.test_questions import TEST_QUESTIONS
from graph.nodes.grade_hallucination import get_hallucination_chain
from graph.nodes.grade_answer import get_answer_chain
from graph.state import GraphState, initial_state
from llm import get_generation_llm, get_grader_llm


# ============================================================
# Baseline pipeline (plain RAG — no self-correction)
# ============================================================
BASELINE_SYSTEM = """You are a question-answering assistant. Answer the question \
using ONLY the provided context. If the context is insufficient, still attempt \
your best answer based on the context."""

BASELINE_HUMAN = """Context:
{context}

Question: {question}

Answer:"""


def _baseline_retrieve(question: str, k: int = 5) -> List[Document]:
    """Plain vector search — no HyDE, no BM25, no reranking."""
    try:
        from retrieval.vector_store import get_vector_store
        return get_vector_store().search(question, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Baseline retrieval failed: {exc}")
        return []


def run_baseline(questions: List[dict]) -> List[dict]:
    """Run plain RAG over each question. No self-correction whatsoever."""
    results = []
    llm = get_generation_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", BASELINE_SYSTEM), ("human", BASELINE_HUMAN)]
    )
    chain = prompt | llm | StrOutputParser()
    hallucination_chain = get_hallucination_chain()
    answer_chain = get_answer_chain()

    for q in questions:
        start = time.time()
        logger.info(f"[BASELINE] {q['id']}: {q['question'][:50]}")
        docs = _baseline_retrieve(q["question"])
        context = "\n\n".join(d.page_content[:1000] for d in docs) or "(no context)"
        try:
            answer = chain.invoke({"context": context, "question": q["question"]})
        except Exception as exc:  # noqa: BLE001
            answer = f"[generation error: {exc}]"

        # Grade hallucination + answer quality (for metrics only; baseline never acts on these).
        try:
            hgrade = hallucination_chain.invoke(
                {"documents": context, "generation": answer}
            )
            grounded = bool(hgrade.grounded)
            hscore = float(hgrade.confidence_contribution)
        except Exception:
            grounded, hscore = True, 0.5
        try:
            agrade = answer_chain.invoke(
                {"question": q["question"], "generation": answer}
            )
            addresses = bool(agrade.addresses)
        except Exception:
            addresses = True

        elapsed = time.time() - start
        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "ground_truth": q["ground_truth"],
            "answer": answer,
            "documents": len(docs),
            "hallucination_free": grounded,
            "hallucination_score": hscore,
            "answer_addresses": addresses,
            "confidence_score": hscore,  # baseline has no real confidence
            "low_confidence": False,      # baseline never flags
            "clarification_needed": False,
            "contradiction_found": False,
            "web_search_used": False,
            "techniques_used": ["Baseline RAG"],
            "processing_time": round(elapsed, 2),
        })
    return results


# ============================================================
# Ultimate pipeline (full self-correction)
# ============================================================
def run_ultimate(questions: List[dict]) -> List[dict]:
    """Run the full self-correcting pipeline over each question."""
    from graph.graph import run_query

    results = []
    for q in questions:
        logger.info(f"[ULTIMATE] {q['id']}: {q['question'][:50]}")
        try:
            state = run_query(q["question"])
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Ultimate pipeline failed on {q['id']}: {exc}")
            state = initial_state(q["question"])
            state["generation"] = f"[pipeline error: {exc}]"

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "ground_truth": q["ground_truth"],
            "answer": state.get("generation", ""),
            "documents": len(state.get("documents", [])),
            "hallucination_free": state.get("hallucination_free", False),
            "hallucination_score": state.get("hallucination_score", 0.0),
            "answer_addresses": state.get("answer_addresses_question", False),
            "confidence_score": state.get("confidence_score", 0.0),
            "low_confidence": state.get("low_confidence", False),
            "clarification_needed": state.get("clarification_needed", False),
            "clarification_question": state.get("clarification_question", ""),
            "contradiction_found": state.get("contradiction_found", False),
            "contradiction_detail": state.get("contradiction_detail", ""),
            "web_search_used": state.get("web_search_used", False),
            "techniques_used": state.get("techniques_used", []),
            "processing_time": state.get("processing_time", 0.0),
        })
    return results


# ============================================================
# Metrics
# ============================================================
def _category_match(result: dict) -> bool:
    """Did the pipeline's behavior match the expected category?

    This is the 'self-correction correctness' check: did the system do the
    RIGHT thing for this kind of question?
    """
    cat = result["category"]
    if cat == "ambiguous":
        return bool(result.get("clarification_needed"))
    if cat == "web":
        return bool(result.get("web_search_used"))
    if cat == "contradictory":
        return bool(result.get("contradiction_found"))
    if cat == "low_confidence":
        return bool(result.get("low_confidence"))
    if cat in ("factual", "multihop"):
        # Should produce a grounded answer, not a clarification/flag.
        return (bool(result.get("hallucination_free"))
                and not result.get("clarification_needed"))
    return False


def compute_metrics(results: List[dict]) -> dict:
    """Aggregate per-run results into metric summary."""
    n = len(results) or 1
    return {
        "n_questions": len(results),
        "hallucination_rate": round(
            sum(1 for r in results if not r.get("hallucination_free")) / n, 3),
        "precision_rate": round(
            sum(1 for r in results if r.get("answer_addresses")) / n, 3),
        "self_correction_rate": round(
            sum(1 for r in results if _category_match(r)) / n, 3),
        "avg_confidence": round(
            sum(r.get("confidence_score", 0.0) for r in results) / n, 3),
        "clarification_rate": round(
            sum(1 for r in results if r.get("clarification_needed")) / n, 3),
        "low_confidence_rate": round(
            sum(1 for r in results if r.get("low_confidence")) / n, 3),
        "web_search_rate": round(
            sum(1 for r in results if r.get("web_search_used")) / n, 3),
        "contradiction_rate": round(
            sum(1 for r in results if r.get("contradiction_found")) / n, 3),
        "avg_processing_time": round(
            sum(r.get("processing_time", 0.0) for r in results) / n, 3),
    }


# ============================================================
# Reporting
# ============================================================
def print_comparison_table(baseline: dict, ultimate: dict) -> None:
    """Pretty-print a before/after comparison."""
    print("\n" + "=" * 72)
    print("  HALLUCINATION & SELF-CORRECTION: BEFORE vs AFTER")
    print("=" * 72)
    fmt = "  {:<28} {:>14} {:>14} {:>14}"
    print(fmt.format("METRIC", "BASELINE", "ULTIMATE", "DELTA"))
    print("  " + "-" * 70)

    rows = [
        ("Hallucination rate", "hallucination_rate", True),    # lower is better
        ("Precision (addrs Q)", "precision_rate", False),       # higher is better
        ("Self-correction rate", "self_correction_rate", False),
        ("Avg confidence", "avg_confidence", False),
        ("Clarification rate", "clarification_rate", False),
        ("Low-confidence rate", "low_confidence_rate", False),
        ("Web search rate", "web_search_rate", False),
        ("Contradiction rate", "contradiction_rate", False),
        ("Avg processing time (s)", "avg_processing_time", True),
    ]
    for label, key, lower_better in rows:
        b = baseline.get(key, 0)
        u = ultimate.get(key, 0)
        delta = u - b
        arrow = "↓" if (delta < 0) == lower_better else "↑"
        good = (delta < 0) == lower_better
        sign = "+" if delta >= 0 else ""
        marker = " ✓" if good and delta != 0 else ("  " if delta == 0 else " ✗")
        print(fmt.format(label, f"{b:.3f}", f"{u:.3f}",
                         f"{sign}{delta:.3f} {arrow}{marker}"))
    print("=" * 72 + "\n")


def print_detail_table(results: List[dict], title: str) -> None:
    """Print per-question results."""
    print(f"\n--- {title} ---")
    print("  {:<4} {:<14} {:<10} {:<10} {:<10} {:<8}".format(
        "ID", "CATEGORY", "GROUNDED", "ADDRS Q", "CONF", "TIME"))
    for r in results:
        print("  {:<4} {:<14} {:<10} {:<10} {:<10} {:<8}".format(
            r["id"], r["category"],
            "yes" if r.get("hallucination_free") else "NO",
            "yes" if r.get("answer_addresses") else "no",
            f"{r.get('confidence_score', 0):.2f}",
            f"{r.get('processing_time', 0):.1f}s",
        ))


# ============================================================
# Main entry point
# ============================================================
def run_evaluation(questions: List[dict] = None, save: bool = True) -> dict:
    """Run baseline + ultimate evaluations and return the comparison."""
    questions = questions or TEST_QUESTIONS
    logger.info(f"=== EVALUATION START: {len(questions)} questions ===")

    print("\n" + "#" * 72)
    print("#  RUNNING BASELINE (plain RAG, no self-correction)")
    print("#" * 72)
    baseline_results = run_baseline(questions)
    baseline_metrics = compute_metrics(baseline_results)

    print("\n" + "#" * 72)
    print("#  RUNNING ULTIMATE (full self-correcting pipeline)")
    print("#" * 72)
    ultimate_results = run_ultimate(questions)
    ultimate_metrics = compute_metrics(ultimate_results)

    print_comparison_table(baseline_metrics, ultimate_metrics)
    print_detail_table(baseline_results, "BASELINE DETAIL")
    print_detail_table(ultimate_results, "ULTIMATE DETAIL")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "baseline_metrics": baseline_metrics,
        "ultimate_metrics": ultimate_metrics,
        "baseline_results": baseline_results,
        "ultimate_results": ultimate_results,
    }

    if save:
        try:
            EVALUATION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
            EVALUATION_RESULTS.write_text(
                json.dumps(output, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info(f"Results saved to {EVALUATION_RESULTS}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not save results: {exc}")

    return output


if __name__ == "__main__":
    run_evaluation()
