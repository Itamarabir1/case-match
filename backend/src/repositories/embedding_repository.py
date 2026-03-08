"""Embedding implementation using sentence-transformers. No business logic."""
from src.infrastructure.embedding_client import embed as _embed


class EmbeddingRepository:
    """IEmbeddingRepository implementation. Delegates to infrastructure."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _embed(texts)
