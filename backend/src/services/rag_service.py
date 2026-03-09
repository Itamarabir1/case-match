"""RAG service: retrieval + Groq LLM analysis."""
import json
import re
import urllib.error
import urllib.request
from collections.abc import Generator
from pathlib import Path

from src.config import get_settings
from src.prompts.rag import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from src.schemas.analyze import RAGAnalysisStructured
from src.schemas.query import SearchQuery
from src.schemas.search_result import RankedCase, SearchResult
from src.services.retrieval_service import RetrievalService
from src.utils.text import safe_filename

MAX_CHARS_PER_CASE = 3500
MAX_CHARS_NEW_CASE = 4000

_JSON_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

# Section headers in raw LLM output (numbered or plain)
_SECTION_LEGAL = re.compile(
    r"(?i)\*\*\s*(?:\d\.\s*)?Legal\s+Pattern\s*\*\*|##\s*Legal\s+Pattern|Legal\s+Pattern\s*:"
)
_SECTION_OUTCOME = re.compile(
    r"(?i)\*\*\s*(?:\d\.\s*)?Common\s+Outcome\s*\*\*|##\s*Common\s+Outcome|Common\s+Outcome\s*:"
)
_SECTION_CONSIDERATIONS = re.compile(
    r"(?i)\*\*\s*(?:\d\.\s*)?Key\s+Considerations\s*\*\*|##\s*Key\s+Considerations|Key\s+Considerations\s*:"
)

# JSON schema for Groq response_format (from Pydantic model)
_GROQ_RAG_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_analysis",
        "strict": False,
        "schema": RAGAnalysisStructured.model_json_schema(),
    },
}


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
    """Return content after start_marker until end_marker or end of text. Content is trimmed."""
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


def _bullet_lines_to_string(content: str) -> str:
    """Extract bullet points (lines starting with -, *, • or digits.) and return newline-separated."""
    return "\n".join(_bullet_lines_to_list(content))


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
    """Parse raw analysis text into RAGAnalysisStructured using section headers. Returns None if sections not found."""
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
    """Parse by splitting on exact markers: **1. Legal Pattern**, **2. Common Outcome**, **3. Key Considerations**."""
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


def _full_text_for_case(case: RankedCase, texts_dir: Path) -> str:
    """Return full text for a case: from exported file if present, else joined snippets."""
    path = texts_dir / f"{safe_filename(case.doc_id)}.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            pass
    return "\n\n".join(case.snippets).strip() if case.snippets else "(no text)"


def get_cases_and_prompts(
    query: str, top_k: int | None = None
) -> tuple[list[RankedCase], str, str]:
    """
    Run retrieval and build context + prompts for Groq (no LLM call).
    Returns (cases, system_prompt, user_prompt). Raises GroqUnavailableError if no cases (caller may still stream).
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    texts_dir = Path(settings.exports_texts_dir)
    request = SearchQuery(query=query.strip(), top_k=k)
    search_result = RetrievalService().search(request)
    cases = search_result.cases
    if not cases:
        return [], RAG_SYSTEM_PROMPT, ""

    context_parts = []
    for i, case in enumerate(cases, 1):
        full = _full_text_for_case(case, texts_dir)
        if len(full) > MAX_CHARS_PER_CASE:
            full = full[:MAX_CHARS_PER_CASE] + "\n[... truncated ...]"
        header = f"--- Case {i} (doc_id={case.doc_id}, {case.score_percent}% similarity)"
        if case.title:
            header += f": {case.title}"
        header += " ---"
        context_parts.append(f"{header}\n{full}")
    context = "\n\n".join(context_parts)
    new_case_trimmed = query.strip()
    if len(new_case_trimmed) > MAX_CHARS_NEW_CASE:
        new_case_trimmed = new_case_trimmed[:MAX_CHARS_NEW_CASE] + "\n[... truncated ...]"
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, new_case=new_case_trimmed)
    return cases, RAG_SYSTEM_PROMPT, user_prompt


def stream_groq_tokens(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    """
    Call Groq with stream=True; yield content tokens as they arrive.
    Raises GroqUnavailableError if GROQ_API_KEY missing; RuntimeError on API errors.
    """
    settings = get_settings()
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        raise GroqUnavailableError(
            "Groq API key is missing. Set GROQ_API_KEY in .env to use Full RAG Analysis."
        )
    url = f"{settings.groq_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "python-requests/2.31.0",
    }
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": True,
    }
    payload["response_format"] = _GROQ_RAG_SCHEMA
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            buffer = ""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    part, buffer = buffer.split("\n\n", 1)
                    for line in part.split("\n"):
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            j = json.loads(data)
                            choices = j.get("choices") or []
                            if choices and isinstance(choices[0], dict):
                                delta = (choices[0] or {}).get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            pass
    except urllib.error.HTTPError as e:
        body_read = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        detail = body_read[:500]
        try:
            err_json = json.loads(body_read)
            if "error" in err_json and isinstance(err_json["error"], dict):
                detail = err_json["error"].get("message", body_read)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Groq API error: {e.code} {e.reason}\n{detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Groq API at {settings.groq_base_url}. Check your internet connection.\n{e.reason}"
        ) from e


def _call_groq(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 300,
    response_format: dict | None = None,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "python-requests/2.31.0",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content") or ""
            return str(content).strip()
        return ""
    except urllib.error.HTTPError as e:
        body_read = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        detail = body_read
        try:
            err_json = json.loads(body_read)
            if "error" in err_json and isinstance(err_json["error"], dict):
                detail = err_json["error"].get("message", body_read)
            elif "error" in err_json and isinstance(err_json["error"], str):
                detail = err_json["error"]
        except json.JSONDecodeError:
            detail = body_read[:500] if body_read else f"{e.code} {e.reason}"
        raise RuntimeError(f"Groq API error: {e.code} {e.reason}\n{detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Groq API at {base_url}. Check your internet connection.\n{e.reason}"
        ) from e


class GroqUnavailableError(Exception):
    """Raised when Groq API key is missing or invalid for /analyze."""

    def __init__(self, message: str = "Groq API is not configured. Set GROQ_API_KEY in .env."):
        self.message = message
        super().__init__(message)


def run_rag(
    query: str, top_k: int | None = None
) -> tuple[list[RankedCase], str, RAGAnalysisStructured | None, str]:
    """
    Run retrieval + Groq analysis.
    Returns (cases, analysis_text, analysis_json, model_name).
    analysis_json is set when the LLM returns valid JSON; otherwise None, analysis_text is raw.
    Raises GroqUnavailableError if GROQ_API_KEY is missing.
    Raises RuntimeError on Groq API or retrieval errors.
    """
    settings = get_settings()
    cases, system_prompt, user_prompt = get_cases_and_prompts(query, top_k)
    if not cases:
        return [], "", None, settings.groq_model
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        raise GroqUnavailableError(
            "Groq API key is missing. Set GROQ_API_KEY in .env to use Full RAG Analysis."
        )
    try:
        raw = _call_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
            response_format=_GROQ_RAG_SCHEMA,
        )
    except RuntimeError as e:
        if "400" in str(e):
            raw = _call_groq(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key,
                base_url=settings.groq_base_url,
                model=settings.groq_model,
            )
        else:
            raise
    # Parse Groq response as JSON into analysis_json; build analysis (raw text) from it for backward compatibility
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
    if structured is not None:
        kc_str = "\n".join(structured.key_considerations) if structured.key_considerations else ""
        analysis_text = (
            f"**Legal Pattern**\n{structured.legal_pattern}\n\n"
            f"**Common Outcome**\n{structured.common_outcome}\n\n"
            f"**Key Considerations**\n{kc_str}"
        )
        if structured.summary:
            analysis_text = f"{structured.summary}\n\n{analysis_text}"
        if structured.caveats:
            analysis_text += "\n\n**Caveats**\n" + "\n".join(f"- {c}" for c in structured.caveats)
    else:
        analysis_text = raw
    # Always return an analysis_json object: use parsed structured or fallback from raw
    if structured is None:
        structured = RAGAnalysisStructured(
            legal_pattern=raw,
            common_outcome="",
            key_considerations=[],
            summary=None,
            caveats=None,
        )
    return cases, analysis_text, structured, settings.groq_model
