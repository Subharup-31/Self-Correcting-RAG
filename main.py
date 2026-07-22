"""
Ultimate Self-Correcting RAG Pipeline — entry point.

Usage:
  python main.py serve            Start the FastAPI server (+ API for the frontend)
  python main.py ingest [path]    Ingest documents (default: ./documents/)
  python main.py query "..."      Run a single query through the full pipeline
  python main.py evaluate         Run the 12-question evaluation harness
  python main.py demo             Generate sample docs, ingest them, run demo queries
  python main.py generate-docs    (Re)generate the sample PDF documents
  python main.py reset            Wipe all indexed documents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DOCUMENTS_DIR
from loguru import logger


# ---------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------- #
def cmd_serve(args):
    """Start the FastAPI server."""
    from api.server import serve
    logger.info(f"Starting API server on {args.host}:{args.port}")
    serve(host=args.host, port=args.port)


def cmd_ingest(args):
    """Ingest documents from a path (file or directory)."""
    from ingestion.pipeline import ingest_directory, ingest_file
    path = args.path or str(DOCUMENTS_DIR)
    p = Path(path)
    if not p.exists():
        logger.error(f"Path not found: {path}")
        sys.exit(1)
    if p.is_dir():
        summary = ingest_directory(path)
    else:
        summary = ingest_file(path)
    print("\n=== INGESTION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_query(args):
    """Run a single query through the full pipeline."""
    from graph.graph import run_query
    state = run_query(args.question)
    print("\n" + "=" * 72)
    print(f"QUERY: {args.question}")
    print("=" * 72)
    print(f"\nANSWER:\n{state.get('generation', '(no answer)')}\n")
    print("-" * 72)
    print(f"Confidence:        {state.get('confidence_score', 0):.2f}")
    print(f"Low confidence:    {state.get('low_confidence', False)}")
    if state.get("confidence_reason"):
        print(f"  reason: {state['confidence_reason']}")
    print(f"Clarification:     {state.get('clarification_needed', False)}")
    if state.get("clarification_question"):
        print(f"  question: {state['clarification_question']}")
    print(f"Contradiction:     {state.get('contradiction_found', False)}")
    if state.get("contradiction_detail"):
        print(f"  detail: {state['contradiction_detail']}")
    print(f"CRAG state:        {state.get('crag_state', '')}")
    print(f"Hallucination-free:{state.get('hallucination_free', False)}")
    print(f"Web search used:   {state.get('web_search_used', False)}")
    print(f"Retries:           {state.get('retry_count', 0)}")
    print(f"Techniques:        {', '.join(state.get('techniques_used', []))}")
    print(f"Processing time:   {state.get('processing_time', 0):.2f}s")
    sources = state.get("sources", [])
    if sources:
        print(f"\nSources ({len(sources)}):")
        for s in sources[:5]:
            print(f"  - {s.get('source', '?')} p.{s.get('page', '?')} "
                  f"[{s.get('doc_type', '?')}]")
    print("=" * 72)


def cmd_evaluate(args):
    """Run the full evaluation harness."""
    from evaluation.harness import run_evaluation
    result = run_evaluation()


def cmd_demo(args):
    """Generate sample docs, ingest them, and run demo queries covering each
    self-correction scenario."""
    print("\n" + "#" * 72)
    print("#  ULTIMATE RAG — DEMO MODE")
    print("#" * 72)

    # 1. Generate sample documents.
    print("\n[1/3] Generating sample documents...")
    from scripts.generate_sample_docs import generate_all
    generate_all(force=args.force)

    # 2. Ingest them.
    print("\n[2/3] Ingesting documents...")
    from ingestion.pipeline import ingest_directory
    summary = ingest_directory(str(DOCUMENTS_DIR))
    print(f"  Ingested: {summary}")

    # 3. Run demo queries — one per scenario.
    print("\n[3/3] Running demo queries...")
    from graph.graph import run_query
    demos = [
        ("FACTUAL", "What is HyDE and how does it improve retrieval?"),
        ("AMBIGUOUS", "Tell me about the main issues."),
        ("CONTRADICTION", "What was the company's revenue in 2024?"),
        ("MULTIHOP", "Compare BM25 and vector search and explain which is better for keyword-heavy queries."),
        ("OUT-OF-SCOPE", "What is today's stock price of Apple?"),
    ]
    for label, q in demos:
        print(f"\n{'─' * 72}")
        print(f"[{label}] {q}")
        print("─" * 72)
        try:
            state = run_query(q)
            if state.get("clarification_needed"):
                print(f"  → CLARIFY: {state['clarification_question']}")
            elif state.get("contradiction_found"):
                print(f"  → CONTRADICTION: {state['contradiction_detail']}")
            else:
                answer = state.get("generation", "(no answer)")
                print(f"  → ANSWER: {answer[:300]}")
            print(f"  confidence={state.get('confidence_score', 0):.2f} "
                  f"low={state.get('low_confidence', False)} "
                  f"web={state.get('web_search_used', False)} "
                  f"techniques={state.get('techniques_used', [])}")
        except Exception as exc:  # noqa: BLE001
            print(f"  → ERROR: {exc}")


def cmd_generate_docs(args):
    """Just generate the sample PDFs."""
    from scripts.generate_sample_docs import generate_all
    generate_all(force=args.force)


def cmd_reset(args):
    """Wipe all stores."""
    from ingestion.pipeline import reset_stores
    reset_stores()
    print("All stores reset.")


# ---------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ultimate-rag",
        description="Ultimate Self-Correcting RAG Pipeline",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_ingest = sub.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("path", nargs="?", default=None,
                          help="File or directory (default: ./documents/)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Run a single query")
    p_query.add_argument("question", help="The question to ask")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("evaluate", help="Run the evaluation harness")
    p_eval.set_defaults(func=cmd_evaluate)

    p_demo = sub.add_parser("demo", help="Generate sample docs, ingest, run demos")
    p_demo.add_argument("--force", action="store_true",
                        help="Regenerate sample docs even if they exist")
    p_demo.set_defaults(func=cmd_demo)

    p_gendocs = sub.add_parser("generate-docs", help="Generate sample PDFs")
    p_gendocs.add_argument("--force", action="store_true")
    p_gendocs.set_defaults(func=cmd_generate_docs)

    p_reset = sub.add_parser("reset", help="Wipe all indexed documents")
    p_reset.set_defaults(func=cmd_reset)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
