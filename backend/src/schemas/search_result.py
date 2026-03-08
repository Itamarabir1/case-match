"""Pydantic schemas for search results. DTOs for API."""
from pydantic import BaseModel, Field, computed_field


class RankedCase(BaseModel):
    """One ranked case (document) – aggregate of chunk scores."""
    doc_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    chunk_count: int = Field(..., ge=0)
    snippets: list[str] = Field(default_factory=list, max_length=10)
    title: str | None = None
    citation: str | None = None
    state: str | None = None
    issuer: str | None = None
    court: str | None = None
    date_filed: str | None = None
    disposition: str | None = None

    @computed_field
    @property
    def score_percent(self) -> int:
        """Similarity as percentage (0–100)."""
        return round(self.score * 100)


class SearchResult(BaseModel):
    """API response – list of ranked cases."""
    query: str
    cases: list[RankedCase] = Field(default_factory=list)
    query_embedding: list[float] | None = None
    best_chunk_id: str | None = None
    best_chunk_score: float | None = None
