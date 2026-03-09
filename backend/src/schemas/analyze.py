"""Pydantic schemas for /analyze endpoint."""
from pydantic import BaseModel, Field, field_validator

from src.schemas.search_result import RankedCase


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""
    query: str = Field(..., min_length=1, description="Legal case description or question")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Number of similar cases to use")


def _to_considerations_list(v: str | list[str]) -> list[str]:
    """Normalize key_considerations to list of strings."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    return [ln.strip() for ln in s.splitlines() if ln.strip()]


class RAGAnalysisStructured(BaseModel):
    """Structured RAG analysis (JSON schema returned by LLM)."""

    legal_pattern: str = Field(
        ..., description="Recurring legal principles or doctrines across the cases"
    )
    common_outcome: str = Field(
        ...,
        description="Typical outcome in similar cases and likely outcome for the new case",
    )
    key_considerations: list[str] = Field(
        ..., description="Practical steps or arguments to strengthen the case (list of items)"
    )
    summary: str | None = Field(default=None, description="Optional one-sentence summary")
    caveats: list[str] | None = Field(
        default=None, description="Optional limitations or disclaimers"
    )

    @field_validator("key_considerations", mode="before")
    @classmethod
    def key_considerations_to_list(cls, v: str | list[str]) -> list[str]:
        return _to_considerations_list(v)


class AnalyzeResponse(BaseModel):
    """Response for POST /analyze."""
    query: str
    cases: list[RankedCase] = Field(default_factory=list)
    analysis: str = Field(default="", description="Full analysis as text (e.g. formatted from structured)")
    analysis_json: RAGAnalysisStructured | None = Field(
        default=None, description="Structured analysis (JSON schema)"
    )
    model: str = ""
