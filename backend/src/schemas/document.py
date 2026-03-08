"""Pydantic schemas for document/case. Validation only."""
from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    """Input for creating a document from raw text."""
    doc_id: str = Field(..., min_length=1, max_length=512)
    text: str = Field(..., min_length=1)


class DocumentIn(BaseModel):
    """Document as stored (minimal)."""
    doc_id: str
    text: str
