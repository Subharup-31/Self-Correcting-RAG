"""
Generate the 4 synthetic sample documents used to test every self-correction scenario.

Documents:
  1. company_report_2024.pdf  — Digital PDF with financial data; INTENTIONAL
     contradictions vs. technical_manual.pdf (different revenue & employee numbers)
  2. research_notes_scanned.pdf — Text rendered to an image and embedded in a PDF,
     so it has NO text layer (forces the OCR path)
  3. technical_manual.pdf     — Dense technical RAG content (HyDE, BM25, CRAG,
     cross-encoders, parent-child chunking) with messy formatting
  4. meeting_minutes.pdf      — Vague, context-dependent content (triggers
     ambiguous CRAG + clarifying questions)

Uses reportlab for digital PDFs and fpdf2/Pillow for the scanned-image PDF.
Falls back gracefully if a library is missing.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from config import DOCUMENTS_DIR


def _build_company_report(path: Path) -> None:
    """Digital PDF with financial data. NOTE: revenue & employees here CONTRADICT
    the technical_manual.pdf to trigger the contradiction detector."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    story = [
        Paragraph("Nexora Technologies — Annual Report 2024", styles["Title"]),
        Spacer(1, 0.3 * inch),
        Paragraph("Financial Highlights", styles["Heading1"]),
        Paragraph(
            "In fiscal year 2024, Nexora Technologies reported total revenue of "
            "<b>$487 million</b>, representing 23% year-over-year growth. The company "
            "ended the year with <b>2,840 employees</b> across 14 global offices.",
            styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Revenue Breakdown", styles["Heading2"]),
        Paragraph(
            "Cloud services contributed $290M (59.5% of revenue), on-premise software "
            "licensing $112M (23.0%), and professional services $85M (17.5%). The "
            "gross margin improved to 68.4% from 64.1% in the prior year.",
            styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Headcount & Operations", styles["Heading2"]),
        Paragraph(
            "As of December 31, 2024, Nexora employed 2,840 full-time staff. The "
            "engineering division accounted for 1,420 employees, sales and marketing "
            "680, and operations 740. Hiring was concentrated in the APAC region.",
            styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Outlook", styles["Heading1"]),
        Paragraph(
            "Management projects 2025 revenue between $590M and $610M, driven by "
            "expanded AI product offerings and the recently announced partnership "
            "with Helix Data Platforms.",
            styles["BodyText"]),
    ]
    doc.build(story)
    print(f"  ✓ {path.name} (digital PDF)")


def _build_technical_manual(path: Path) -> None:
    """Dense technical RAG content. NOTE: revenue & employees here CONTRADICT the
    company_report_2024.pdf (different numbers) to test contradiction detection.
    Also covers the factual RAG/AI questions (HyDE, cross-encoders, etc.)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    # Use a mix of font sizes / headings to simulate "inconsistent formatting".
    story = [
        Paragraph("Nexora Internal Technical Manual v3.2", styles["Title"]),
        Spacer(1, 0.25 * inch),
        Paragraph("Section 1 — Company Snapshot", styles["Heading1"]),
        Paragraph(
            "Nexora Technologies closed FY2024 with <b>$512 million</b> in revenue "
            "and a global headcount of <b>3,150 employees</b>. (Note: these are the "
            "audited internal figures and differ from the preliminary numbers in the "
            "external annual report.)",
            styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Section 2 — Retrieval-Augmented Generation (RAG)", styles["Heading1"]),
        Paragraph(
            "RAG combines a retriever (which fetches relevant documents) with a "
            "generator (which produces an answer conditioned on those documents). "
            "Standard RAG retrieves once and generates once; it has no mechanism to "
            "detect whether retrieval succeeded.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph("2.1 HyDE — Hypothetical Document Embeddings", styles["Heading2"]),
        Paragraph(
            "HyDE asks the LLM to first generate a HYPOTHETICAL answer to the query, "
            "then uses that hypothetical document as the search query. Because "
            "answer-to-answer semantic matching is more precise than question-to-answer "
            "matching, HyDE improves retrieval recall on abstract questions. The "
            "hypothetical document is discarded after retrieval — only the real "
            "retrieved documents reach the generator.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph("2.2  Cross-Encoder Reranking", styles["Heading2"]),
        Paragraph(
            "A cross-encoder scores (query, document) pairs jointly using full "
            "attention across both texts. This is far more accurate than bi-encoder "
            "similarity (used for initial vector search) but slower, because every "
            "(query, doc) pair requires a separate forward pass. The standard pattern "
            "is two-stage: a fast bi-encoder retrieves candidates, then a cross-encoder "
            "reranks the top-K.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph("2.3  Parent-Child Chunking", styles["Heading2"]),
        Paragraph(
            "Parent-child chunking splits each document into large PARENT chunks "
            "(e.g. 1024 tokens) and small CHILD chunks (e.g. 256 tokens). Children "
            "are embedded and retrieved for precise matching; when a child matches, "
            "the system returns the full PARENT as generation context. This gives "
            "the LLM coherent, broader context than the tiny retrieved chunk alone.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph("2.4  BM25 vs Vector Search", styles["Heading2"]),
        Paragraph(
            "BM25 is a lexical (keyword) retriever: it scores documents by term "
            "frequency and inverse document frequency. It excels when queries share "
            "exact vocabulary with documents (names, IDs, code, jargon). Vector "
            "search embeds both query and documents into a semantic space and matches "
            "by cosine similarity; it excels when meaning matters more than wording. "
            "For keyword-heavy queries — exact product names, error codes, technical "
            "identifiers — BM25 typically outperforms vector search. Hybrid retrieval "
            "fuses both via Reciprocal Rank Fusion (RRF) to get the strengths of each.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph("2.5  CRAG — Corrective RAG", styles["Heading2"]),
        Paragraph(
            "CRAG adds a self-correction layer to RAG. After retrieval, each document "
            "is GRADED for relevance. If the retrieval is judged poor, CRAG falls "
            "back to web search before generating. Standard RAG has no such check — "
            "it generates from whatever was retrieved, even if irrelevant. Use CRAG "
            "when retrieval quality is uncertain or the corpus is noisy; use plain "
            "RAG when the corpus is clean and queries reliably match.",
            styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Section 3 — Formatting Notes (intentionally messy)", styles["Heading3"]),
        Paragraph("This manual mixes  font   sizes, irregular spacing,  and "
                  "inconsistent   heading levels to stress-test the document loader.",
                  styles["BodyText"]),
    ]
    doc.build(story)
    print(f"  ✓ {path.name} (digital PDF, messy formatting)")


def _build_research_notes_scanned(path: Path) -> None:
    """Image-only PDF (no text layer). Forces the OCR path in DocumentLoader.

    We render text to a PIL Image and embed it as a full-page image in a PDF,
    so pymupdf's get_text() returns nothing and pytesseract OCR is required.
    """
    from PIL import Image, ImageDraw, ImageFont
    import fitz  # PyMuPDF

    text_blocks = [
        "Research Notes — Self-Improving RAG",
        "",
        "Experiment 7: Confidence Calibration",
        "",
        "We observed that the hallucination grader, when given a binary yes/no",
        "output, tends to be over-confident: it marks 89% of answers as grounded",
        "even when human raters disagree on 34% of them. Switching to a graded",
        "0.0-1.0 confidence score and surfacing unsupported claims explicitly",
        "reduced the silent-hallucination rate from 27% to 4% on the eval set.",
        "",
        "Experiment 8: Contradiction Surfacing",
        "",
        "When two retrieved documents disagree, picking either silently produces",
        "confident-wrong answers. Surfacing the contradiction ('Doc A says X,",
        "Doc B says Y') and letting the user resolve it eliminated an entire",
        "class of failures at the cost of a small clarification-rate increase.",
        "",
        "Experiment 9: Re-Query Budget",
        "",
        "Allowing up to 3 retrieval retries with query rewriting recovered",
        "useful context in 41% of initially-poor retrievals. Beyond 3 retries",
        "the marginal gain dropped below 3% — not worth the latency cost.",
    ]

    # Render to an image.
    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font = ImageFont.load_default()
    y = 120
    for line in text_blocks:
        draw.text((120, y), line, fill="black", font=font)
        y += 50

    # Build a PDF with this image as the only page (no text layer).
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US letter
    rect = fitz.Rect(0, 0, 612, 792)
    page.insert_image(rect, stream=img_byte_arr.read())
    doc.save(str(path))
    doc.close()
    print(f"  ✓ {path.name} (scanned image PDF — requires OCR)")


def _build_meeting_minutes(path: Path) -> None:
    """Vague, context-dependent content. Triggers ambiguous CRAG + clarifying
    questions (the topics are unfocused and reference unspecified 'it', 'the
    issue', etc.)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    story = [
        Paragraph("Meeting Minutes — Project Sync", styles["Title"]),
        Spacer(1, 0.25 * inch),
        Paragraph("Attendees: the team.", styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph(
            "We discussed the main issues. It was generally agreed that it needs "
            "more work before we can move forward. Several people raised concerns "
            "about it, and we decided to revisit it next week.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph(
            "Regarding the other thing, the recommendation is to handle it the "
            "usual way. We will check in on it at the next standup and see if it "
            "improved.",
            styles["BodyText"]),
        Spacer(1, 0.15 * inch),
        Paragraph(
            "Action items: someone will look into it and report back. We should "
            "also make sure it is documented properly so it does not happen again.",
            styles["BodyText"]),
    ]
    doc.build(story)
    print(f"  ✓ {path.name} (vague content — triggers clarification)")


def generate_all(force: bool = False) -> list[Path]:
    """Generate all 4 sample documents. Returns list of created paths."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("company_report_2024.pdf", _build_company_report),
        ("technical_manual.pdf", _build_technical_manual),
        ("research_notes_scanned.pdf", _build_research_notes_scanned),
        ("meeting_minutes.pdf", _build_meeting_minutes),
    ]
    created = []
    print(f"Generating sample documents in {DOCUMENTS_DIR} ...")
    for name, builder in targets:
        path = DOCUMENTS_DIR / name
        if path.exists() and not force:
            print(f"  - {name} already exists, skipping (use force=True to rebuild)")
            created.append(path)
            continue
        try:
            builder(path)
            created.append(path)
        except ImportError as exc:
            print(f"  ✗ {name}: missing dependency ({exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name}: {exc}")
    return created


if __name__ == "__main__":
    force = "--force" in sys.argv
    generate_all(force=force)
