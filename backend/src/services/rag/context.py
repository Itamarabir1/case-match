"""Build RAG context: retrieval + prompt assembly (no LLM call)."""
from pathlib import Path

from src.config import get_settings
from src.prompts.rag import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from src.schemas.query import SearchQuery
from src.schemas.search_result import RankedCase
from src.services.retrieval_service import RetrievalService
from src.utils.text import safe_filename

MAX_CHARS_PER_CASE = 3500
MAX_CHARS_NEW_CASE = 4000


def full_text_for_case(case: RankedCase, texts_dir: Path) -> str:
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
    Returns (cases, system_prompt, user_prompt). Caller may still stream cases when empty.
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
        full = full_text_for_case(case, texts_dir)
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
