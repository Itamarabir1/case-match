"""RAG service: facade for backward compatibility. Implementation lives in src.services.rag."""
from src.services.rag import (
    GroqUnavailableError,
    get_cases_and_prompts,
    run_rag,
    stream_groq_tokens,
)

__all__ = [
    "GroqUnavailableError",
    "get_cases_and_prompts",
    "run_rag",
    "stream_groq_tokens",
]
