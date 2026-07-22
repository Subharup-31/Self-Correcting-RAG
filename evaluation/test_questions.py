"""
12 test questions for the evaluation harness.

Categories cover every self-correction scenario:
  - factual       → should answer from docs
  - ambiguous     → should trigger a clarifying question
  - web           → should route to / use web search (real-time info)
  - contradictory → should flag the contradiction between docs
  - multihop      → should trigger query decomposition
  - low_confidence→ should be flagged LOW_CONFIDENCE (insufficient docs)

Questions relate to RAG/AI concepts that will appear in the generated sample
documents (company_report_2024.pdf, technical_manual.pdf, etc.).
"""

TEST_QUESTIONS = [
    # === Factual (should answer cleanly from docs) ===
    {
        "id": "F1",
        "question": "What is HyDE and how does it improve retrieval?",
        "ground_truth": "HyDE (Hypothetical Document Embeddings) generates a "
                        "hypothetical answer to the query and uses it for retrieval, "
                        "improving semantic match between query and documents.",
        "category": "factual",
    },
    {
        "id": "F2",
        "question": "What does the cross-encoder reranker do in the RAG pipeline?",
        "ground_truth": "The cross-encoder reranker scores (query, document) pairs "
                        "jointly to refine the order of retrieved documents for higher precision.",
        "category": "factual",
    },
    {
        "id": "F3",
        "question": "What is the purpose of parent-child chunking?",
        "ground_truth": "Parent-child chunking uses small child chunks for precise "
                        "retrieval and larger parent chunks as generation context.",
        "category": "factual",
    },

    # === Ambiguous (should trigger a clarifying question) ===
    {
        "id": "A1",
        "question": "Tell me about the main issues.",
        "ground_truth": "CLARIFICATION_NEEDED",
        "category": "ambiguous",
    },
    {
        "id": "A2",
        "question": "How does it work?",
        "ground_truth": "CLARIFICATION_NEEDED",
        "category": "ambiguous",
    },

    # === Out-of-scope / real-time (should use web search) ===
    {
        "id": "W1",
        "question": "What is today's stock price of Apple?",
        "ground_truth": "WEB_SEARCH",
        "category": "web",
    },
    {
        "id": "W2",
        "question": "What were the major news headlines today?",
        "ground_truth": "WEB_SEARCH",
        "category": "web",
    },

    # === Contradictory (should flag the contradiction explicitly) ===
    {
        "id": "C1",
        "question": "What was the company's revenue in 2024?",
        "ground_truth": "CONTRADICTION_FLAGGED",
        "category": "contradictory",
    },
    {
        "id": "C2",
        "question": "How many employees does the company have?",
        "ground_truth": "CONTRADICTION_FLAGGED",
        "category": "contradictory",
    },

    # === Multi-hop (triggers decomposition) ===
    {
        "id": "M1",
        "question": "Compare BM25 and vector search and explain which is better for keyword-heavy queries.",
        "ground_truth": "BM25 excels at keyword-heavy queries; vector search is better "
                        "for semantic matching. A hybrid approach combines both.",
        "category": "multihop",
    },
    {
        "id": "M2",
        "question": "What is the difference between CRAG and standard RAG, and when should you use each?",
        "ground_truth": "CRAG adds self-correction (grading retrieved docs, web fallback) "
                        "to standard RAG; use CRAG when retrieval quality is uncertain.",
        "category": "multihop",
    },

    # === Low-confidence (insufficient docs) ===
    {
        "id": "L1",
        "question": "What is the airspeed velocity of an unladen swallow?",
        "ground_truth": "LOW_CONFIDENCE",
        "category": "low_confidence",
    },
]


def get_questions_by_category(category: str) -> list:
    """Filter test questions by category."""
    return [q for q in TEST_QUESTIONS if q["category"] == category]


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(q["category"] for q in TEST_QUESTIONS)
    print(f"Total questions: {len(TEST_QUESTIONS)}")
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")
