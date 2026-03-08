"""Interface for vector store – data access abstraction."""
from typing import Protocol

from src.schemas.chunk import ChunkIn, ChunkOut


class IVectorStoreRepository(Protocol):
    """Vector store: add chunks and query by vector."""

    def add_chunks(self, chunks: list[ChunkIn], embeddings: list[list[float]]) -> None:
        """Persist chunks and their embeddings."""
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ChunkOut]:
        """Return top_k chunks by similarity. Scores in [0, 1]."""
        ...
