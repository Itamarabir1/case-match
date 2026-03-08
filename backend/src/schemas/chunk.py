"""Pydantic schemas for chunks. Validation only."""
from pydantic import BaseModel, Field


class ChunkIn(BaseModel):
    """Single chunk with metadata for storage."""
    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    doc_meta: dict[str, str] | None = None


class ChunkOut(BaseModel):
    """Chunk as returned from retrieval (with score)."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: float = Field(..., ge=0.0, le=1.0)
    title: str | None = None
    citation: str | None = None
    state: str | None = None
    issuer: str | None = None
    court: str | None = None
    date_filed: str | None = None
    disposition: str | None = None
