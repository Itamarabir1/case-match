"""Chroma client – external integration. No business logic."""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings

_client = None


def get_chroma_client():
    """Return singleton persistent Chroma client. Creates path if needed."""
    global _client
    if _client is None:
        settings = get_settings()
        path = Path(settings.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client
