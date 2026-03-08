"""Pydantic schemas for search query. Validation only."""
from pydantic import BaseModel, Field

from src.config.constants import MAX_QUERY_LENGTH_CHARS


class SearchQuery(BaseModel):
    """Search request – legal problem description."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH_CHARS,
        description="Legal problem or question in text",
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
