"""Chroma client – external integration. No business logic."""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings


def get_chroma_client():
    """Return a persistent Chroma client. Creates path if needed."""
    settings = get_settings()
    path = Path(settings.chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
