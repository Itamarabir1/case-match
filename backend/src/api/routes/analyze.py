"""Analyze API: POST /analyze – RAG retrieval + Groq analysis (streaming SSE)."""
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.schemas.analyze import AnalyzeRequest
from src.services.rag_service import (
    GroqUnavailableError,
    get_cases_and_prompts,
    stream_groq_tokens,
)

router = APIRouter(tags=["analyze"])


def _sse_event(data: dict) -> str:
    """Format a dict as one SSE event line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/analyze")
def analyze_endpoint(body: AnalyzeRequest):
    """Stream retrieval results (cases) then Groq analysis tokens as SSE."""
    def generate():
        try:
            cases, system_prompt, user_prompt = get_cases_and_prompts(body.query, body.top_k)
            cases_payload = [c.model_dump() for c in cases]
            yield _sse_event({"type": "cases", "cases": cases_payload})
            if not cases:
                yield _sse_event({"type": "done"})
                return
            if not user_prompt.strip():
                yield _sse_event({"type": "done"})
                return
            for token in stream_groq_tokens(system_prompt, user_prompt):
                yield _sse_event({"type": "token", "content": token})
            yield _sse_event({"type": "done"})
        except GroqUnavailableError as e:
            yield _sse_event({"type": "error", "message": e.message})
        except RuntimeError as e:
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
