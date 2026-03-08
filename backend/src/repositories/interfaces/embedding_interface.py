"""Interface for embedding – data access abstraction."""
from typing import Protocol


class IEmbeddingRepository(Protocol):
    """Embedding: encode texts to vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return list of embedding vectors for given texts."""
        ...
