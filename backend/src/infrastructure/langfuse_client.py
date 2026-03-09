"""Langfuse client for LLM observability. Returns None when disabled or keys are missing."""
from langfuse import Langfuse

from src.config import get_settings

_client = None


def get_langfuse_client() -> Langfuse | None:
    """Return a singleton Langfuse client, or None if disabled or keys not set."""
    global _client
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    key = (settings.langfuse_public_key or "").strip()
    if not key:
        return None
    if _client is None:
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _client
