# 🚀 Ultimate Self-Correcting RAG Pipeline

A production-ready RAG system over **messy, unstructured documents** (mixed PDFs, scanned images needing OCR, inconsistent formatting) that **detects when its own retrieved context is insufficient or contradictory** — and either **re-queries, asks a clarifying question, or explicitly flags low-confidence answers** instead of hallucinating.

This project is a merger of three reference RAG implementations:
- **Self-Healing-RAG** → pipeline structure (HyDE, CRAG, cross-encoder reranker, dynamic few-shot)
- **Self-Corrective-Agentic-RAG** → retrieval (BM25 + vector + RRF hybrid, knowledge strips)
- **Agentic-Adaptive-RAG** → quality graders (LangGraph hallucination grader, answer grader, router)

---

## ✨ Key features

| Capability | How it works |
|---|---|
| **Messy document ingestion** | PyMuPDF for digital PDFs, automatic pytesseract OCR fallback for scanned pages, confidence score recorded per chunk |
| **Hybrid retrieval** | BM25 (lexical) + bge-m3 vector search, fused with Reciprocal Rank Fusion (RRF, k=60) |
| **HyDE** | Generate a hypothetical answer → embed → retrieve real docs (better semantic match) |
| **3-state CRAG grading** | Each retrieved doc graded `correct` / `ambiguous` / `incorrect` |
| **Clarifying questions** | When context is ambiguous, the system ASKS instead of guessing |
| **Contradiction detection** | Pairwise check of top docs; conflicts surfaced explicitly in the answer |
| **Re-query loop** | Poor retrieval → rewrite query → re-retrieve (up to 3×) → fall back to web |
| **Hallucination grading** | Generated answer checked for grounding; specific unsupported claims listed |
| **Confidence scoring** | Weighted aggregate of all quality signals; answers below 0.5 are FLAGGED |
| **Dynamic few-shot learning** | Thumbs-up feedback stored and retrieved for similar future queries |
| **Evaluation harness** | 12 test questions run baseline-vs-ultimate, outputs hallucination-rate table |

---

## 🏗 Architecture

```
Entry → route_question
  ├─ "direct_llm" → generate → END
  ├─ "websearch"  → web_search → retrieve → grade_documents → ...
  └─ "vectorstore" → decompose → retrieve (hybrid+HyDE) → grade_documents
                                                     │
                            ┌────────────────────────┼────────────────────────┐
                        "correct"               "ambiguous"              "incorrect"
                            │                       │                        │
                 detect_contradiction        clarify (ask Q)        query_rewrite → retrieve
                            │                      END                (max 3 retries, then web)
                     rerank (cross-encoder)
                            │
                     few-shot inject
                            │
                       generate
                            │
                     grade_hallucination ── not grounded ──► regenerate (max 3)
                            │ grounded
                     confidence_scorer
                            │
                     ┌──────┴──────┐
                 low_conf       high_conf
                     │             │
                END (flag)    grade_answer
                                   │
                            ┌──────┴──────┐
                       not useful     useful
                            │           │
                       web_search      END
```

---

## 📁 Folder structure

```
Ultimate-RAG/
├── config.py                  # Central configuration
├── llm.py                     # LLM factory (Gemini by default)
├── main.py                    # CLI entry point
├── ingestion/
│   ├── document_loader.py     # PDF/Image/Word/HTML loader + OCR
│   ├── chunking.py            # Parent-child chunking
│   └── pipeline.py            # Ingestion orchestrator
├── retrieval/
│   ├── vector_store.py        # ChromaDB + bge-m3 embeddings
│   ├── bm25_retriever.py      # BM25 keyword retrieval
│   ├── hybrid_retriever.py    # RRF fusion
│   ├── hyde.py                # HyDE
│   └── reranker.py            # Cross-encoder reranker
├── graph/
│   ├── state.py               # GraphState TypedDict
│   ├── graph.py               # Master LangGraph assembly
│   ├── nodes/                 # 11 graph nodes
│   └── chains/                # Query decomposer, few-shot learner
├── evaluation/
│   ├── test_questions.py      # 12 test questions
│   └── harness.py             # Baseline-vs-ultimate harness
├── api/
│   └── server.py              # FastAPI + SSE
├── frontend/                  # React + Vite + TypeScript
├── scripts/
│   └── generate_sample_docs.py
├── documents/                 # Sample PDFs (auto-generated)
├── chroma_db/                 # ChromaDB persistent storage
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick start

### 1. Install dependencies

```bash
cd "e:/self improving rag/Ultimate-RAG"
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

**System dependencies for OCR:**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed and on your PATH.
- For PDF rendering, PyMuPDF (`fitz`) is pip-installable.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set at least GOOGLE_API_KEY (free at https://aistudio.google.com/)
# TAVILY_API_KEY is optional but enables web-search fallback (https://tavily.com/)
```

### 3. Generate sample documents & ingest

```bash
python main.py generate-docs       # Creates 4 sample PDFs in documents/
python main.py ingest              # Loads + chunks + indexes them
```

### 4. Try it

```bash
# Run a single factual query
python main.py query "What is HyDE and how does it improve retrieval?"

# Run the full demo (generates docs, ingests, runs one query per scenario)
python main.py demo

# Run the 12-question evaluation harness (baseline vs ultimate)
python main.py evaluate
```

### 5. Start the server + frontend

```bash
# Backend
python main.py serve               # FastAPI at http://localhost:8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                        # Vite at http://localhost:5173
```

---

## 🧪 Evaluation

The harness runs 12 questions twice — once as **plain RAG** (vector search → generate, no self-correction), once as the **full self-correcting pipeline** — and reports:

| Metric | Meaning |
|---|---|
| `hallucination_rate` | % answers not grounded in retrieved docs |
| `precision_rate` | % answers that address the question |
| `self_correction_rate` | % cases where the right correction mechanism fired |
| `avg_confidence` | Mean confidence score |
| `clarification_rate` | % questions that triggered a clarifying question |
| `low_confidence_rate` | % answers flagged as low confidence |
| `web_search_rate` | % questions that used web fallback |
| `contradiction_rate` | % questions that surfaced a contradiction |

Example output:

```
  METRIC                        BASELINE       ULTIMATE         DELTA
  ----------------------------------------------------------------------
  Hallucination rate               0.417          0.083       -0.333 ↓ ✓
  Precision (addrs Q)              0.583          0.833       +0.250 ↑ ✓
  Self-correction rate             0.000          0.750       +0.750 ↑ ✓
  Avg confidence                   0.500          0.712       +0.212 ↑ ✓
  ...
```

Results are saved to `evaluation/results.json`.

---

## 🔌 API reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload & ingest documents (multipart) |
| `POST` | `/api/query` | Run a query through the full pipeline |
| `POST` | `/api/feedback` | Submit thumbs up/down for few-shot learning |
| `GET` | `/api/statistics` | Store sizes + few-shot learner stats |
| `GET` | `/api/evaluate` | Run the 12-question evaluation harness |
| `GET` | `/api/health` | Health check (API key presence, store readiness) |
| `DELETE` | `/api/documents` | Clear all indexed documents |
| `GET` | `/api/query/stream?q=...` | SSE stream of the live pipeline trace |

`POST /api/query` response:

```json
{
  "query": "What was the company's revenue in 2024?",
  "answer": "Note: the sources disagree — [company_report_2024.pdf] says $487M...",
  "confidence_score": 0.42,
  "low_confidence": true,
  "clarification_needed": false,
  "contradiction_found": true,
  "contradiction_detail": "[company_report_2024.pdf] vs [technical_manual.pdf]: ...",
  "crag_state": "correct",
  "hallucination_free": true,
  "web_search_used": false,
  "sources": [{"source": "company_report_2024.pdf", "page": 1, "excerpt": "..."}],
  "techniques_used": ["Routing", "HyDE", "CRAG (3-state grading)", "Contradiction Detection", ...],
  "processing_time": 4.21,
  "retry_count": 0
}
```

---

## 🎨 Frontend

A dark, glassmorphism React + Vite + TypeScript UI with 4 tabs:

- **Chat** — ask questions; each answer shows a confidence badge (green/yellow/red), low-confidence warning, clarification dialog, contradiction alert, collapsible pipeline trace, and source citations. Thumbs up/down for feedback.
- **Documents** — drag-and-drop upload, ingested document list with type/chunk count.
- **Pipeline Trace** — live SSE stream of node activations on a visual graph.
- **Evaluation Dashboard** — run the harness, bar chart of hallucination rate before/after.

```bash
cd frontend
npm install
npm run dev
```

---

## ⚙️ Configuration

All tunables live in `config.py` and can be overridden via environment variables (`.env`):

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (required) |
| `TAVILY_API_KEY` | — | Tavily key for web-search fallback (optional) |
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `LLM_MODEL` | `gemini-1.5-flash` | Model name |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Local embedding model |
| `CONFIDENCE_THRESHOLD` | `0.5` | Below this → low-confidence flag |
| `MAX_RETRY_COUNT` | `3` | Max re-query / regeneration iterations |

---

## 🧰 Tech stack

- **LLM**: Google Gemini 1.5 Flash (`langchain-google-genai`, free tier)
- **Embeddings**: BAAI/bge-m3 (`sentence-transformers`, 1024-dim, local)
- **Vector store**: ChromaDB (local persistent)
- **Orchestration**: LangGraph
- **BM25**: rank_bm25
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Web search**: Tavily (`langchain-tavily`)
- **OCR**: pytesseract + PyMuPDF
- **API**: FastAPI + uvicorn + SSE
- **Frontend**: React + Vite + TypeScript
- **Evaluation**: ragas (optional)

---

## 🔍 Sample documents

`python main.py generate-docs` creates 4 synthetic PDFs in `documents/`:

1. **`company_report_2024.pdf`** — Digital PDF. Revenue **$487M**, employees **2,840**.
2. **`technical_manual.pdf`** — Dense RAG/AI technical content; revenue **$512M**, employees **3,150** (intentional contradiction with #1).
3. **`research_notes_scanned.pdf`** — Image-only PDF (no text layer) → forces the OCR path.
4. **`meeting_minutes.pdf`** — Vague content ("it", "the issue") → triggers ambiguous CRAG + clarifying questions.

---

## 📝 License

MIT — built for educational and research purposes.
