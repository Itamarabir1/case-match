"""Analyze API: POST /analyze – RAG retrieval + Groq analysis (SSE: cases then full analysis)."""
import json
import threading
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.config import get_settings
from src.infrastructure.langfuse_client import get_langfuse_client
from src.schemas.analyze import AnalyzeRequest
from src.services.rag.groq_client import GROQ_RAG_SCHEMA, GroqUnavailableError, call_groq
from src.services.rag_service import get_cases_and_prompts

router = APIRouter(tags=["analyze"])

# Design: Groq Structured Outputs (response_format) do not support streaming.
# We use a single call_groq() and send the full response in one SSE event.


def _sse_event(data: dict) -> str:
    """Format a dict as one SSE event line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _flush_langfuse_in_background(langfuse):
    """Run flush in a daemon thread so it never blocks the request."""
    if langfuse is None:
        return
    def _run():
        try:
            langfuse.flush()
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start()


@router.post("/analyze")
def analyze_endpoint(body: AnalyzeRequest):
    """
    Return cases then Groq analysis as SSE.
    Uses non-streaming Groq call (Structured Outputs are not streamable). Langfuse flush runs in background.
    """
    settings = get_settings()

    def generate():
        # Contract: every path yields valid SSE; any exception becomes an error event.
        try:
            yield from _generate_body(body, settings)
        except Exception as e:
            yield _sse_event({"type": "error", "message": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _generate_body(body: AnalyzeRequest, settings):
    """Yield SSE events. Raises on failure; caller turns exceptions into error events."""
    try:
        cases, system_prompt, user_prompt = get_cases_and_prompts(body.query, body.top_k)
    except Exception:
        raise
    try:
            cases_payload = [c.model_dump() for c in cases]
            yield _sse_event({"type": "cases", "cases": cases_payload})
            if not cases:
                yield _sse_event({"type": "done"})
                return
            if not user_prompt.strip():
                yield _sse_event({"type": "done"})
                return
            api_key = (settings.groq_api_key or "").strip()
            if not api_key:
                yield _sse_event({"type": "error", "message": "Groq API key is missing. Set GROQ_API_KEY in .env."})
                return
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            start = time.time()
            full_text = ""
            usage = {}
            langfuse = get_langfuse_client()
            try:
                if langfuse:
                    with langfuse.start_as_current_observation(
                        name="analyze",
                        as_type="span",
                        input={"query": body.query, "cases_count": len(cases)},
                    ):
                        with langfuse.start_as_current_observation(
                            name="groq-completion",
                            as_type="generation",
                            model=settings.groq_model,
                            input=messages,
                        ) as gen:
                            try:
                                full_text, usage = call_groq(
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    api_key=api_key,
                                    base_url=settings.groq_base_url,
                                    model=settings.groq_model,
                                    response_format=GROQ_RAG_SCHEMA,
                                )
                            except RuntimeError as e:
                                if "400" in str(e):
                                    full_text, usage = call_groq(
                                        system_prompt=system_prompt,
                                        user_prompt=user_prompt,
                                        api_key=api_key,
                                        base_url=settings.groq_base_url,
                                        model=settings.groq_model,
                                    )
                                else:
                                    raise
                            usage_details = None
                            if isinstance(usage.get("prompt_tokens"), int) or isinstance(usage.get("completion_tokens"), int):
                                usage_details = {
                                    "prompt_tokens": usage.get("prompt_tokens", 0),
                                    "completion_tokens": usage.get("completion_tokens", 0),
                                }
                            gen.update(output=full_text, usage_details=usage_details)
                    _flush_langfuse_in_background(langfuse)
                else:
                    try:
                        full_text, usage = call_groq(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            api_key=api_key,
                            base_url=settings.groq_base_url,
                            model=settings.groq_model,
                            response_format=GROQ_RAG_SCHEMA,
                        )
                    except RuntimeError as e:
                        if "400" in str(e):
                            full_text, usage = call_groq(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                api_key=api_key,
                                base_url=settings.groq_base_url,
                                model=settings.groq_model,
                            )
                        else:
                            raise
            except Exception:
                _flush_langfuse_in_background(langfuse)
                raise
            duration_ms = int((time.time() - start) * 1000)
            # Send full analysis in one event (frontend accumulates and parses on done)
            if full_text:
                yield _sse_event({"type": "token", "content": full_text})
            done_payload = {
                "type": "done",
                "duration_ms": duration_ms,
                "model": settings.groq_model,
            }
            if usage:
                done_payload["usage"] = usage
            yield _sse_event(done_payload)
        except GroqUnavailableError as e:
            yield _sse_event({"type": "error", "message": e.message})
        except RuntimeError as e:
            yield _sse_event({"type": "error", "message": str(e)})
        except Exception as e:
            yield _sse_event({"type": "error", "message": str(e)})
