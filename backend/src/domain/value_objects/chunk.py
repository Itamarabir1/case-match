"""Value object: chunk. No framework dependencies."""
from dataclasses import dataclass


@dataclass
class Chunk:
    """Value object for a text chunk with metadata."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
