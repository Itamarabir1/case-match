"""Pydantic schemas for /analyze endpoint."""
from pydantic import BaseModel, Field

from src.schemas.search_result import RankedCase


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""
    query: str = Field(..., min_length=1, description="Legal case description or question")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Number of similar cases to use")


class AnalyzeResponse(BaseModel):
    """Response for POST /analyze."""
    query: str
    cases: list[RankedCase] = Field(default_factory=list)
    analysis: str = ""
    model: str = ""
