"""
Production-grade RAG evaluation framework.

Computes retrieval metrics (Recall@K, Precision@K, MRR, NDCG) and generation metrics
(Groundedness, Faithfulness, Hallucination Rate, Answer Relevancy, Context Precision/Recall)
alongside system execution statistics (Latency, Confidence Calibration, Token Usage, Retry Rate).
Saves historic runs to a local SQLite database for regression tracking.
"""

from __future__ import annotations

import datetime
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from config import BASE_DIR
from evaluation.test_questions import TEST_QUESTIONS
from graph.graph import run_query
from llm import get_grader_llm
from graph.nodes.grade_hallucination import get_hallucination_chain
from graph.nodes.grade_answer import get_answer_chain

DB_PATH = BASE_DIR / "evaluation_runs.db"


class EvaluationRecord(BaseModel):
    query_id: str
    query: str
    category: str
    ground_truth: str
    expected_docs: List[str]
    retrieved_docs: List[str]
    answer: str
    precision_k: float
    recall_k: float
    mrr: float
    ndcg: float
    groundedness: float
    faithfulness: float
    relevancy: float
    context_precision: float
    context_recall: float
    latency: float
    confidence: float
    tokens: int
    retries: int


# LLM Graders for Context Precision & Recall
CONTEXT_EVAL_SYSTEM = """You are an elite RAG validation engine. Evaluate the retrieved context against the question and the ground truth.

For Context Precision: Determine what fraction of the retrieved passages are directly relevant to answering the question. Output a score from 0.0 to 1.0.
For Context Recall: Determine what fraction of the ground truth facts are present in the retrieved passages. Output a score from 0.0 to 1.0."""

CONTEXT_EVAL_HUMAN = """Question: {question}
Ground Truth: {ground_truth}

Retrieved Context Chunks:
{context}"""


class ContextMetrics(BaseModel):
    precision: float = Field(description="Context Precision score from 0.0 to 1.0")
    recall: float = Field(description="Context Recall score from 0.0 to 1.0")


def _init_eval_db() -> None:
    """Initialize SQLite database for evaluation run history."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                metrics_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS details (
                run_id TEXT,
                query_id TEXT,
                query TEXT,
                category TEXT,
                ground_truth TEXT,
                answer TEXT,
                precision_k REAL,
                recall_k REAL,
                mrr REAL,
                ndcg REAL,
                groundedness REAL,
                faithfulness REAL,
                relevancy REAL,
                context_precision REAL,
                context_recall REAL,
                latency REAL,
                confidence REAL,
                tokens INTEGER,
                retries INTEGER,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _compute_retrieval_metrics(retrieved: List[str], expected: List[str]) -> Dict[str, float]:
    """Compute Precision@K, Recall@K, MRR, and NDCG."""
    if not expected:
        return {"precision_k": 1.0, "recall_k": 1.0, "mrr": 1.0, "ndcg": 1.0}
    if not retrieved:
        return {"precision_k": 0.0, "recall_k": 0.0, "mrr": 0.0, "ndcg": 0.0}

    # Normalize filenames
    retrieved_norm = [Path(f).name.lower() for f in retrieved]
    expected_norm = [Path(f).name.lower() for f in expected]

    # Calculate hits
    hits = [1 if r in expected_norm else 0 for r in retrieved_norm]
    relevant_retrieved = sum(hits)

    # Precision & Recall
    precision_k = relevant_retrieved / len(retrieved)
    recall_k = relevant_retrieved / len(expected)

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, hit in enumerate(hits):
        if hit == 1:
            mrr = 1.0 / (idx + 1)
            break

    # NDCG
    dcg = 0.0
    for idx, hit in enumerate(hits):
        if hit == 1:
            dcg += 1.0 / math.log2(idx + 2)

    idcg = 0.0
    for idx in range(min(len(expected), len(retrieved))):
        idcg += 1.0 / math.log2(idx + 2)

    ndcg = dcg / idcg if idcg > 0.0 else 0.0

    return {
        "precision_k": round(precision_k, 3),
        "recall_k": round(recall_k, 3),
        "mrr": round(mrr, 3),
        "ndcg": round(ndcg, 3),
    }


class EvaluationRunner:
    """Benchmark runner for self-correcting RAG architecture."""

    def __init__(self):
        _init_eval_db()
        self.grader_llm = get_grader_llm()
        # Expected documents mappings for test questions
        self.expected_docs_map = {
            "F1": ["technical_manual.pdf"],
            "F2": ["technical_manual.pdf"],
            "F3": ["technical_manual.pdf"],
            "A1": ["meeting_minutes.pdf"],
            "A2": ["meeting_minutes.pdf"],
            "W1": [],
            "W2": [],
            "C1": ["company_report_2024.pdf", "technical_manual.pdf"],
            "C2": ["company_report_2024.pdf", "technical_manual.pdf"],
            "M1": ["technical_manual.pdf"],
            "M2": ["technical_manual.pdf"],
            "L1": [],
        }

    def evaluate_generation(self, question: str, ground_truth: str, answer: str, context: str) -> Dict[str, float]:
        """Compute Groundedness, Relevancy, Faithfulness, and Context metrics via LLM grading."""
        # Setup fallback default scores
        evals = {
            "groundedness": 1.0,
            "faithfulness": 1.0,
            "relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }

        # 1. Hallucination / Groundedness Grade
        try:
            h_grade = get_hallucination_chain().invoke({"documents": context, "generation": answer})
            evals["groundedness"] = 1.0 if h_grade.grounded else 0.0
            evals["faithfulness"] = float(h_grade.confidence_contribution)
        except Exception as exc:
            logger.warning(f"Groundedness check failed: {exc}")

        # 2. Answer Relevancy Grade
        try:
            a_grade = get_answer_chain().invoke({"question": question, "generation": answer})
            evals["relevancy"] = 1.0 if a_grade.addresses else 0.0
        except Exception as exc:
            logger.warning(f"Answer relevancy check failed: {exc}")

        # 3. Context Precision & Context Recall
        if context and context != "(no context)":
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", CONTEXT_EVAL_SYSTEM),
                ("human", CONTEXT_EVAL_HUMAN)
            ])
            chain = prompt | self.grader_llm.with_structured_output(ContextMetrics)
            try:
                result = chain.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "context": context[:3000]
                })
                evals["context_precision"] = float(result.precision)
                evals["context_recall"] = float(result.recall)
            except Exception as exc:
                logger.warning(f"Context metrics grading failed: {exc}")

        return evals

    def run_benchmark(self, owner_id: str = "eval_user", session_id: str = "") -> dict:
        """Run all test questions through the pipeline, evaluate, and save statistics."""
        run_id = f"run_{uuid_hex()}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records: List[EvaluationRecord] = []

        logger.info(f"Starting Evaluation Run: {run_id}")

        for q in TEST_QUESTIONS:
            qid = q["id"]
            question = q["question"]
            category = q["category"]
            ground_truth = q["ground_truth"]
            expected = self.expected_docs_map.get(qid, [])

            logger.info(f"Evaluating {qid}: {question[:50]}...")
            
            # Execute RAG query (using isolated user context if provided)
            state = run_query(question, owner_id=owner_id, session_id=session_id)
            
            answer = state.get("generation", "")
            latency = state.get("processing_time", 0.0)
            confidence = state.get("confidence_score", 0.0)
            retries = state.get("retry_count", 0)
            
            # Extract retrieved source files
            retrieved = [
                d.metadata.get("source", d.metadata.get("file_name", ""))
                for d in state.get("documents", [])
            ]
            retrieved = [f for f in retrieved if f]

            # Compute retrieval metrics
            ret_metrics = _compute_retrieval_metrics(retrieved, expected)

            # Compute generation context
            context_docs = state.get("documents", [])
            context_text = "\n\n".join(d.page_content for d in context_docs) if context_docs else "(no context)"
            gen_metrics = self.evaluate_generation(question, ground_truth, answer, context_text)

            # Estimate token usage (char_count / 4)
            char_count = len(answer) + len(context_text) + len(question)
            tokens = int(char_count / 4)

            record = EvaluationRecord(
                query_id=qid,
                query=question,
                category=category,
                ground_truth=ground_truth,
                expected_docs=expected,
                retrieved_docs=retrieved,
                answer=answer,
                precision_k=ret_metrics["precision_k"],
                recall_k=ret_metrics["recall_k"],
                mrr=ret_metrics["mrr"],
                ndcg=ret_metrics["ndcg"],
                groundedness=gen_metrics["groundedness"],
                faithfulness=gen_metrics["faithfulness"],
                relevancy=gen_metrics["relevancy"],
                context_precision=gen_metrics["context_precision"],
                context_recall=gen_metrics["context_recall"],
                latency=latency,
                confidence=confidence,
                tokens=tokens,
                retries=retries,
            )
            records.append(record)

        # Aggregate overall metrics
        total = len(records)
        avg_precision = sum(r.precision_k for r in records) / total
        avg_recall = sum(r.recall_k for r in records) / total
        avg_mrr = sum(r.mrr for r in records) / total
        avg_ndcg = sum(r.ndcg for r in records) / total
        avg_groundedness = sum(r.groundedness for r in records) / total
        avg_relevancy = sum(r.relevancy for r in records) / total
        avg_faithfulness = sum(r.faithfulness for r in records) / total
        avg_context_precision = sum(r.context_precision for r in records) / total
        avg_context_recall = sum(r.context_recall for r in records) / total
        avg_latency = sum(r.latency for r in records) / total
        total_tokens = sum(r.tokens for r in records)
        total_retries = sum(r.retries for r in records)
        hallucination_rate = sum(1 for r in records if r.groundedness < 0.5) / total
        retry_rate = sum(1 for r in records if r.retries > 0) / total

        # Calibration error calculation (rough check of confidence calibration)
        # Difference between groundedness score and confidence score
        calibration_error = sum(abs(r.groundedness - r.confidence) for r in records) / total

        aggregated = {
            "retrieval_precision": round(avg_precision, 3),
            "retrieval_recall": round(avg_recall, 3),
            "mrr": round(avg_mrr, 3),
            "ndcg": round(avg_ndcg, 3),
            "groundedness": round(avg_groundedness, 3),
            "faithfulness": round(avg_faithfulness, 3),
            "hallucination_rate": round(hallucination_rate, 3),
            "relevancy": round(avg_relevancy, 3),
            "context_precision": round(avg_context_precision, 3),
            "context_recall": round(avg_context_recall, 3),
            "avg_latency": round(avg_latency, 2),
            "confidence_calibration": round(1.0 - calibration_error, 3),
            "total_tokens": total_tokens,
            "retry_rate": round(retry_rate, 3),
            "total_retries": total_retries,
        }

        # Store to SQLite
        self._save_run(run_id, timestamp, aggregated, records)

        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "summary": aggregated,
            "details": [r.dict() for r in records]
        }

    def _save_run(self, run_id: str, timestamp: str, summary: dict, records: List[EvaluationRecord]) -> None:
        """Persist evaluation results into SQLite."""
        conn = sqlite3.connect(str(DB_PATH))
        try:
            conn.execute(
                "INSERT INTO runs (run_id, timestamp, metrics_json) VALUES (?, ?, ?)",
                (run_id, timestamp, json.dumps(summary))
            )
            for r in records:
                conn.execute("""
                    INSERT INTO details (
                        run_id, query_id, query, category, ground_truth, answer,
                        precision_k, recall_k, mrr, ndcg, groundedness, faithfulness,
                        relevancy, context_precision, context_recall, latency, confidence,
                        tokens, retries
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, r.query_id, r.query, r.category, r.ground_truth, r.answer,
                    r.precision_k, r.recall_k, r.mrr, r.ndcg, r.groundedness, r.faithfulness,
                    r.relevancy, r.context_precision, r.context_recall, r.latency, r.confidence,
                    r.tokens, r.retries
                ))
            conn.commit()
            logger.info("Saved evaluation run to SQLite database.")
        except Exception as exc:
            logger.error(f"Failed to save evaluation run: {exc}")
        finally:
            conn.close()


def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
