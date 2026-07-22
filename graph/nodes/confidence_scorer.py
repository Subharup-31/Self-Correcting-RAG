"""
Confidence scorer node.

Aggregates all quality signals into a single 0.0-1.0 confidence score:

  score = 0.40 * hallucination_score
        + 0.25 * crag_score            (correct=1.0, ambiguous=0.5, incorrect=0.0)
        + 0.20 * answer_addresses_score (1.0 if True else 0.0)
        + 0.15 * retry_penalty          (1.0 - retries/MAX, clamped >= 0)

If score < CONFIDENCE_THRESHOLD (0.5):
  - low_confidence = True
  - Answer is FLAGGED with a human-readable reason (not silently returned).
"""

from __future__ import annotations

from loguru import logger

from config import SelfCorrectionConfig
from graph.state import GraphState

CONFIDENCE_SCORER = "confidence_scorer"


def _retry_penalty(retry_count: int) -> float:
    """1.0 at zero retries, linearly decaying to 0.0 at MAX_RETRY_COUNT."""
    max_r = max(SelfCorrectionConfig.MAX_RETRY_COUNT, 1)
    return max(0.0, 1.0 - (retry_count / max_r))


def compute_confidence(state: GraphState) -> tuple[float, str]:
    """Return (confidence_score, reason_if_low)."""
    cfg = SelfCorrectionConfig

    hallucination_score = float(state.get("hallucination_score", 0.5))
    crag_state = state.get("crag_state", "ambiguous") or "ambiguous"
    crag_score = cfg.CRAG_STATE_SCORES.get(crag_state, 0.5)
    answer_addresses = 1.0 if state.get("answer_addresses_question", False) else 0.0
    retry_count = int(state.get("retry_count", 0))
    retry_pen = _retry_penalty(retry_count)

    score = (
        cfg.WEIGHT_HALLUCINATION * hallucination_score
        + cfg.WEIGHT_CRAG * crag_score
        + cfg.WEIGHT_ANSWER * answer_addresses
        + cfg.WEIGHT_RETRY * retry_pen
    )
    score = max(0.0, min(1.0, score))

    # Build a reason if low.
    reason = ""
    if score < cfg.CONFIDENCE_THRESHOLD:
        parts = []
        if hallucination_score < 0.5:
            parts.append(f"answer weakly grounded in sources ({hallucination_score:.2f})")
        if crag_state == "ambiguous":
            parts.append("retrieved context was ambiguous")
        elif crag_state == "incorrect":
            parts.append("retrieved context was largely irrelevant")
        if not state.get("answer_addresses_question", False):
            parts.append("answer may not address the question")
        if retry_count > 0:
            parts.append(f"required {retry_count} retrieval retries")
        reason = "; ".join(parts) or "overall quality signals below threshold"
    return score, reason


def confidence_scorer(state: GraphState) -> GraphState:
    """Compute the final confidence score and flag low-confidence answers."""
    score, reason = compute_confidence(state)
    low = score < SelfCorrectionConfig.CONFIDENCE_THRESHOLD

    logger.info(
        f"---CONFIDENCE: {score:.2f} low={low} "
        f"{'(' + reason + ')' if reason else ''}---"
    )

    techniques = list(state.get("techniques_used", []))
    if "Confidence Scoring" not in techniques:
        techniques.append("Confidence Scoring")

    return {
        "confidence_score": round(score, 3),
        "low_confidence": low,
        "confidence_reason": reason,
        "techniques_used": techniques,
    }


def decide_after_confidence(state: GraphState) -> str:
    """Conditional edge: if low confidence, end with a flag; else grade answer."""
    if state.get("low_confidence", False):
        return "end_flagged"
    return "grade_answer"
