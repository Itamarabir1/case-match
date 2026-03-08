"""RAG: retrieve 5 similar cases, build context + new case, send to LLM for analysis."""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "backend"))
# Run from project root so .env and relative paths are correct
os.chdir(_project_root)

# Force .env to override system env vars so this script always uses the key from .env
from dotenv import load_dotenv

_env_path = _project_root / ".env"
load_dotenv(_env_path, override=True)

# Debug: what was loaded from .env (avoid caching confusion)
_env_key_raw = os.environ.get("GROQ_API_KEY") or ""
def _mask_key(key: str) -> str:
    if not key or len(key) <= 14:
        return "(empty or too short)" if not key else f"{key[:4]}...{key[-2:]}" if len(key) > 6 else "***"
    return f"{key[:10]}...{key[-4:]}"

if _env_key_raw:
    print(f"[DEBUG] .env loaded from: {_env_path}")
    print(f"[DEBUG] GROQ_API_KEY (from os.environ after load_dotenv): {_mask_key(_env_key_raw)} (len={len(_env_key_raw)})")

from src.config import get_settings
from src.schemas.query import SearchQuery
from src.schemas.search_result import RankedCase
from src.services.retrieval_service import RetrievalService

PROMPT_TEMPLATE = """You are a knowledgeable legal assistant. Your task is to analyze a new court case by comparing it to similar past cases. Use the context from the similar cases to provide a detailed and structured analysis.

Similar court cases:
{context}

New case:
{new_case}

Please provide your analysis in a clear and structured way, covering the following points:

1. **Similarity Analysis**: Explain why these cases are similar to the new case.
2. **Legal Pattern**: Identify any recurring legal patterns, principles, or arguments that appear in these cases.
3. **Common Outcome**: Summarize the outcomes in most of these cases and indicate what is likely to happen in the new case based on this pattern.

Format your response clearly with headings for each section."""


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
    # Debug: exact request (key masked)
    print(f"[DEBUG] Groq URL: {url}")
    print(f"[DEBUG] Groq headers: Content-Type={headers['Content-Type']}, Authorization=Bearer {_mask_key(api_key)}")
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
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
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
        # Must be before URLError: HTTPError is a subclass of URLError
        body = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        detail = body
        try:
            err_json = json.loads(body)
            if "error" in err_json and isinstance(err_json["error"], dict):
                detail = err_json["error"].get("message", body)
            elif "error" in err_json and isinstance(err_json["error"], str):
                detail = err_json["error"]
        except json.JSONDecodeError:
            detail = body[:500] if body else f"{e.code} {e.reason}"
        hint = ""
        if e.code == 403:
            hint = "\n(403: API key invalid or expired? Create a new key at console.groq.com and set GROQ_API_KEY in .env)"
        raise RuntimeError(
            f"Groq API error: {e.code} {e.reason}\n{detail}{hint}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Groq API at {base_url}. Check your internet connection.\n{e.reason}"
        ) from e


# Set to a string to test with a hardcoded key (isolate .env/reading issues). Example: "gsk_xxxx..."
HARDCODED_TEST_KEY: str | None = None


def _run_minimal_groq_test() -> None:
    """Minimal Groq API call with hardcoded URL and optional hardcoded key to isolate 403."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    key = HARDCODED_TEST_KEY if HARDCODED_TEST_KEY else (get_settings().groq_api_key or "").strip()
    if not key:
        print("No key: set GROQ_API_KEY in .env or set HARDCODED_TEST_KEY in this script.")
        sys.exit(1)
    print("[MINIMAL TEST] Groq API isolation test")
    print(f"[MINIMAL TEST] URL: {url}")
    print(f"[MINIMAL TEST] Key (masked): {_mask_key(key)} (len={len(key)}), source={'HARDCODED_TEST_KEY' if HARDCODED_TEST_KEY else '.env/config'}")
    body = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "python-requests/2.31.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print(f"[MINIMAL TEST] Success. Response: {content[:200]}")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore") if e.fp else ""
        print(f"[MINIMAL TEST] Failed: {e.code} {e.reason}")
        print(f"[MINIMAL TEST] Body: {err_body[:500]}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[MINIMAL TEST] Failed (network): {e.reason}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG: analyze a new case using 5 similar cases and an LLM (Groq API).",
    )
    parser.add_argument(
        "query_or_file",
        nargs="?",
        default=None,
        help="New case text or path to .txt file.",
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Run minimal Groq API test (hardcoded URL) and exit. Use HARDCODED_TEST_KEY in script to test with a fixed key.",
    )
    args = parser.parse_args()

    if args.test_api:
        _run_minimal_groq_test()
        return

    settings = get_settings()
    top_k = settings.top_k
    texts_dir = Path(settings.exports_texts_dir)
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        print("Missing GROQ_API_KEY in .env. Add it, then rerun.")
        sys.exit(1)

    if not args.query_or_file:
        # Default: use examples/new_case.txt (relative to project root)
        project_root = Path(__file__).resolve().parents[1]
        default_file = project_root / "examples" / "new_case.txt"
        if default_file.exists() and default_file.is_file():
            new_case = default_file.read_text(encoding="utf-8", errors="ignore").strip()
            print(f"Using default file: {default_file}")
        else:
            new_case = input("Enter new case text or path to .txt file: ").strip()
    else:
        new_case = args.query_or_file.strip()

    p = Path(new_case)
    if p.exists() and p.is_file() and p.suffix.lower() in (".txt",):
        new_case = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not new_case:
            print("File is empty.")
            sys.exit(1)

    if not new_case:
        print("Empty input.")
        sys.exit(1)

    request = SearchQuery(query=new_case, top_k=top_k)
    service = RetrievalService()
    try:
        result = service.search(request)
    except Exception as exc:
        print(f"Search failed: {exc}")
        sys.exit(1)

    cases = result.cases
    if not cases:
        print("No similar cases found. Build the index first (e.g. rebuild_first_courtlistener_cases.py).")
        sys.exit(1)

    # Groq on_demand tier has low TPM; cap context size so request fits (~6k tokens)
    MAX_CHARS_PER_CASE = 3500
    MAX_CHARS_NEW_CASE = 4000
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
    new_case_trimmed = new_case[:MAX_CHARS_NEW_CASE] + ("\n[... truncated ...]" if len(new_case) > MAX_CHARS_NEW_CASE else "")

    prompt = PROMPT_TEMPLATE.format(context=context, new_case=new_case_trimmed)

    # Debug: exact value being used (first 10 + last 4 chars only)
    print(f"[DEBUG] GROQ_API_KEY from config (masked): {_mask_key(api_key)} (len={len(api_key)})")
    print(f"Calling LLM (Groq) [key length={len(api_key)}, model={settings.groq_model}]...")
    try:
        response = _call_groq(
            prompt=prompt,
            api_key=api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RAG Analysis")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
