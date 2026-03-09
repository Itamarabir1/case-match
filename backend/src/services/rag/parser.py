"""Parse raw LLM output into RAGAnalysisStructured (JSON, section headers, or fallback)."""
import json
import re

from src.schemas.analyze import RAGAnalysisStructured

_JSON_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

_SECTION_LEGAL = re.compile(
    r"(?i)\*\*\s*(?:\d\.\s*)?Legal\s+Pattern\s*\*\*|##\s*Legal\s+Pattern|Legal\s+Pattern\s*:"
)
_SECTION_OUTCOME = re.compile(
    r"(?i)\*\*\s*(?:\d\.\s*)?Common\s+Outcome\s*\*\*|##\s*Common\s+Outcome|Common\s+Outcome\s*:"
)
_SECTION_CONSIDERATIONS = re.compile(
    r"(?i)\*\*\s*(?:\d\.\s*)?Key\s+Considerations\s*\*\*|##\s*Key\s+Considerations|Key\s+Considerations\s*:"
)


def _parse_analysis_json(raw: str) -> RAGAnalysisStructured | None:
    """Parse LLM response as JSON into RAGAnalysisStructured. Returns None on failure."""
    s = raw.strip()
    m = _JSON_BLOCK_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        data = json.loads(s)
        return RAGAnalysisStructured.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_section(text: str, start_marker: re.Pattern, end_marker: re.Pattern | None) -> str:
    """Return content after start_marker until end_marker or end of text."""
    m = start_marker.search(text)
    if not m:
        return ""
    start = m.end()
    if end_marker is None:
        return text[start:].strip()
    end_m = end_marker.search(text, start)
    if not end_m:
        return text[start:].strip()
    return text[start : end_m.start()].strip()


def _bullet_lines_to_list(content: str) -> list[str]:
    """Extract bullet points as list of strings."""
    lines = [ln.strip() for ln in content.splitlines()]
    bullets = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        stripped = re.sub(r"^[\-\*•]\s*", "", ln)
        stripped = re.sub(r"^\d+\.\s*", "", stripped)
        if stripped:
            bullets.append(stripped)
    return bullets if bullets else ([content.strip()] if content.strip() else [])


def _parse_analysis_from_raw_text(raw: str) -> RAGAnalysisStructured | None:
    """Parse raw analysis text using section headers. Returns None if sections not found."""
    text = raw.strip()
    legal = _extract_section(text, _SECTION_LEGAL, _SECTION_OUTCOME)
    outcome = _extract_section(text, _SECTION_OUTCOME, _SECTION_CONSIDERATIONS)
    considerations_raw = _extract_section(text, _SECTION_CONSIDERATIONS, None)
    considerations_list = _bullet_lines_to_list(considerations_raw) if considerations_raw else []
    if not considerations_list and considerations_raw:
        considerations_list = [considerations_raw.strip()]
    if not legal.strip() or not outcome.strip() or not considerations_list:
        return None
    return RAGAnalysisStructured(
        legal_pattern=legal.strip(),
        common_outcome=outcome.strip(),
        key_considerations=considerations_list,
        summary=None,
        caveats=None,
    )


def _parse_analysis_by_exact_splits(raw: str) -> RAGAnalysisStructured | None:
    """Parse by splitting on **1. Legal Pattern**, **2. Common Outcome**, **3. Key Considerations**."""
    text = raw.strip()
    markers = [
        "**1. Legal Pattern**",
        "**2. Common Outcome**",
        "**3. Key Considerations**",
    ]
    for m in markers:
        if m.lower() not in text.lower():
            return None
    parts = re.split(
        r"\*\*1\.\s*Legal\s+Pattern\s*\*\*|\*\*2\.\s*Common\s+Outcome\s*\*\*|\*\*3\.\s*Key\s+Considerations\s*\*\*",
        text,
        flags=re.IGNORECASE,
        maxsplit=3,
    )
    if len(parts) < 4:
        return None
    legal = parts[1].strip() if len(parts) > 1 else ""
    outcome = parts[2].strip() if len(parts) > 2 else ""
    considerations_raw = parts[3].strip() if len(parts) > 3 else ""
    considerations_list = _bullet_lines_to_list(considerations_raw) if considerations_raw else []
    if not considerations_list and considerations_raw:
        considerations_list = [considerations_raw.strip()]
    if not legal or not outcome or not considerations_list:
        return None
    return RAGAnalysisStructured(
        legal_pattern=legal,
        common_outcome=outcome,
        key_considerations=considerations_list,
        summary=None,
        caveats=None,
    )


def parse_rag_response(raw: str) -> RAGAnalysisStructured:
    """
    Parse raw LLM output into RAGAnalysisStructured.
    Tries JSON, then exact splits, then section headers. Returns a fallback object if all fail.
    """
    structured = _parse_analysis_json(raw)
    if structured is not None:
        required_filled = (
            (structured.legal_pattern or "").strip()
            and (structured.common_outcome or "").strip()
            and (structured.key_considerations or [])
        )
        if not required_filled:
            structured = None
    if structured is None:
        structured = _parse_analysis_by_exact_splits(raw)
    if structured is None:
        structured = _parse_analysis_from_raw_text(raw)
    if structured is None:
        structured = RAGAnalysisStructured(
            legal_pattern=raw,
            common_outcome="",
            key_considerations=[],
            summary=None,
            caveats=None,
        )
    return structured


def format_analysis_text(structured: RAGAnalysisStructured) -> str:
    """Turn RAGAnalysisStructured into display text (Legal Pattern, Common Outcome, Key Considerations, etc.)."""
    kc_str = "\n".join(structured.key_considerations) if structured.key_considerations else ""
    text = (
        f"**Legal Pattern**\n{structured.legal_pattern}\n\n"
        f"**Common Outcome**\n{structured.common_outcome}\n\n"
        f"**Key Considerations**\n{kc_str}"
    )
    if structured.summary:
        text = f"{structured.summary}\n\n{text}"
    if structured.caveats:
        text += "\n\n**Caveats**\n" + "\n".join(f"- {c}" for c in structured.caveats)
    return text
