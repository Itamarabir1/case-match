"""Domain entity: document (case). No framework dependencies."""
from dataclasses import dataclass


@dataclass
class Document:
    """Domain entity representing a legal document/case."""
    doc_id: str
    text: str
