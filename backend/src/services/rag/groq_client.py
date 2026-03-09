"""Groq API: streaming and non-streaming chat completions for RAG.

Design: Groq does not support streaming when using response_format (Structured Outputs).
Use call_groq() for JSON-schema responses; stream_groq_tokens() only when not using response_format.
"""
import json
import urllib.error
import urllib.request
from collections.abc import Generator

from src.config import get_settings
from src.schemas.analyze import RAGAnalysisStructured

GROQ_RAG_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_analysis",
        "strict": False,
        "schema": RAGAnalysisStructured.model_json_schema(),
    },
}


class GroqUnavailableError(Exception):
    """Raised when Groq API key is missing or invalid for /analyze."""

    def __init__(self, message: str = "Groq API is not configured. Set GROQ_API_KEY in .env."):
        self.message = message
        super().__init__(message)


def stream_groq_tokens(
    system_prompt: str,
    user_prompt: str,
    usage_out: dict | None = None,
) -> Generator[str, None, None]:
    """
    Call Groq with stream=True; yield content tokens as they arrive.
    If usage_out is provided, it may be filled with usage from the last stream chunk (e.g. usage_out["usage"]).
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
    payload["response_format"] = GROQ_RAG_SCHEMA
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
                            if usage_out is not None:
                                usage = j.get("usage")
                                if not usage and isinstance(j.get("x_groq"), dict):
                                    usage = (j.get("x_groq") or {}).get("usage")
                                if isinstance(usage, dict):
                                    usage_out["usage"] = usage
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


def call_groq(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 300,
    response_format: dict | None = None,
) -> tuple[str, dict]:
    """
    Call Groq (non-streaming). Returns (content_str, usage_dict).
    usage_dict has prompt_tokens, completion_tokens, total_tokens when present.
    """
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
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content") or ""
            return str(content).strip(), usage
        return "", usage
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
