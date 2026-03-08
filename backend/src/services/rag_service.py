"""RAG service: retrieval + Groq LLM analysis. Reuses logic from scripts/rag_analysis.py."""
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from src.config import get_settings
from src.schemas.query import SearchQuery
from src.schemas.search_result import RankedCase, SearchResult
from src.services.retrieval_service import RetrievalService

PROMPT_TEMPLATE = """You are a knowledgeable legal assistant. Analyze the following similar court cases and the new case description. Use the context from the similar cases to provide a clear, structured analysis.

Similar court cases:
{context}

New case (user's situation):
{new_case}

Provide your analysis with exactly these three sections (use these headings):

1. **Legal Pattern**
What legal pattern do these cases share? Identify recurring principles, arguments, or doctrines that appear across these cases.

2. **Common Outcome**
What happened in most of these cases? Summarize the typical outcome and what is likely to happen in the new case based on this pattern.

3. **Key Considerations**
What should the plaintiff (or the person in the new case) focus on to strengthen their case? Practical steps or arguments to emphasize."""

MAX_CHARS_PER_CASE = 3500
MAX_CHARS_NEW_CASE = 4000


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "doc"


def _full_text_for_case(case: RankedCase, texts_dir: Path) -> str:
    """Return full text for a case: from exported file if present, else joined snippets."""
    path = texts_dir / f"{_safe_name(case.doc_id)}.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            pass
    return "\n\n".join(case.snippets).strip() if case.snippets else "(no text)"


def _call_groq(prompt: str, api_key: str, base_url: str, model: str, timeout: int = 300) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "python-requests/2.31.0",
    }
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledgeable legal assistant. Answer clearly and with the requested headings.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
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


def run_rag(query: str, top_k: int | None = None) -> tuple[list[RankedCase], str, str]:
    """
    Run retrieval + Groq analysis. Returns (cases, analysis_text, model_name).
    Raises GroqUnavailableError if GROQ_API_KEY is missing.
    Raises RuntimeError on Groq API or retrieval errors.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    texts_dir = Path(settings.exports_texts_dir)
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        raise GroqUnavailableError(
            "Groq API key is missing. Set GROQ_API_KEY in .env to use Full RAG Analysis."
        )

    request = SearchQuery(query=query.strip(), top_k=k)
    search_result = RetrievalService().search(request)
    cases = search_result.cases
    if not cases:
        return [], "", settings.groq_model

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

    prompt = PROMPT_TEMPLATE.format(context=context, new_case=new_case_trimmed)
    analysis = _call_groq(
        prompt=prompt,
        api_key=api_key,
        base_url=settings.groq_base_url,
        model=settings.groq_model,
    )
    return cases, analysis, settings.groq_model
