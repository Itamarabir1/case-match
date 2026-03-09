"""Cases API: GET /cases/{doc_id}/text – full case text as PDF (opens inline in browser)."""
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from fpdf import FPDF

from src.config import get_settings

router = APIRouter(prefix="/cases", tags=["cases"])

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Citation parentheses: (Doc. # ...), (Id. at ...), (Dkt. 39), (ECF No. 5), etc.
_CITATION_PATTERN = re.compile(
    r"\s*\([^)]*(?:Doc(?:s)?\.\s*#|Id\.\s+at)[^)]*\)",
    re.IGNORECASE,
)
_CITATION_LIKE_PATTERN = re.compile(
    r"\s*\((?=[^)]*(?:Doc(?:s)?\.?|Id\.?|\bDkt\.\s*\d|\bECF\s+No\.?\s*\d|\bat\s+[p¶]\.?|\b[p¶]\.\s*\d|see\s+))[^)]{1,200}\)",
    re.IGNORECASE,
)
# Bracket notations: [ECF No. 1], [Dkt. 1], [Doc. 5], etc.
_BRACKET_CITATION = re.compile(
    r"\s*\[(?:ECF\s+No\.?\s*\d+|Dkt\.\s*\d+|Doc(?:s)?\.?\s*#?\s*\d+[^\]]*)\]",
    re.IGNORECASE,
)
# Footnote/section markers (remove or replace with space)
_FOOTNOTE_MARKERS = re.compile(r"[¶§]")

# 2cm in mm
_MARGIN_MM = 20


def _clean_case_text(raw: str) -> str:
    """Remove citations, bracket refs, footnote markers; normalize line breaks."""
    text = _CITATION_PATTERN.sub(" ", raw)
    text = _CITATION_LIKE_PATTERN.sub(" ", text)
    text = _BRACKET_CITATION.sub(" ", text)
    text = _FOOTNOTE_MARKERS.sub(" ", text)
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)  # collapse multiple spaces
    return text.strip()


def _sanitize_pdf_text(s: str) -> str:
    """Ensure text is safe for Helvetica (replace non-Latin-1)."""
    return s.encode("latin-1", errors="replace").decode("latin-1")


class CasePDF(FPDF):
    def __init__(self, doc_id: str) -> None:
        super().__init__()
        self.doc_id = doc_id
        self.set_auto_page_break(auto=True, margin=_MARGIN_MM)
        self.set_margins(_MARGIN_MM, _MARGIN_MM, _MARGIN_MM)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", style="I", size=8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _build_case_pdf(doc_id: str, cleaned_text: str) -> bytes:
    """Generate PDF bytes: title, subtitle, body with paragraph spacing, page numbers."""
    pdf = CasePDF(doc_id)
    pdf.add_page()
    pdf.alias_nb_pages("{nb}")

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, _sanitize_pdf_text(f"Case {doc_id}"), ln=True)
    pdf.ln(2)
    # Subtitle (doc_id)
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, _sanitize_pdf_text(f"Document ID: {doc_id}"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Body: paragraphs with spacing
    pdf.set_font("Helvetica", size=11)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_text) if p.strip()]
    for para in paragraphs:
        safe = _sanitize_pdf_text(para)
        pdf.multi_cell(0, 6, safe)
        pdf.ln(3)  # paragraph spacing

    return bytes(pdf.output())


@router.get("/{doc_id}/text")
def get_case_text(doc_id: str) -> Response:
    """Return case full text as a PDF (opens inline in browser)."""
    if not doc_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    settings = get_settings()
    texts_dir = Path(settings.exports_texts_dir)
    if not texts_dir.is_absolute():
        texts_dir = _BACKEND_ROOT / texts_dir
    path = texts_dir / f"{doc_id}.txt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Case text not found locally")
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = _clean_case_text(raw)
    pdf_bytes = _build_case_pdf(doc_id, cleaned)
    filename = f"case_{doc_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )
