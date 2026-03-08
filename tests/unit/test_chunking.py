"""Unit test: chunking utility."""
import pytest

from src.schemas.chunk import ChunkIn
from src.utils.chunking import split_into_chunks


def test_split_into_chunks_returns_list_of_chunk_in() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = split_into_chunks("doc1", text, chunk_size=20, chunk_overlap=5)
    assert isinstance(chunks, list)
    assert all(isinstance(c, ChunkIn) for c in chunks)
    assert all(c.doc_id == "doc1" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_split_into_chunks_empty_text_returns_empty() -> None:
    chunks = split_into_chunks("doc1", "   ")
    assert chunks == []
