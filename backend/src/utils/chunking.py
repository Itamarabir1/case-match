"""Chunking utility – RecursiveCharacterTextSplitter. Pure function."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.config.constants import CHUNK_SEPARATORS
from src.schemas.chunk import ChunkIn


def split_into_chunks(
    doc_id: str,
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
    doc_meta: dict[str, str] | None = None,
) -> list[ChunkIn]:
    """Split text into chunks. Returns list of ChunkIn for storage."""
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    seps = separators or CHUNK_SEPARATORS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=seps,
        length_function=len,
    )
    chunks = splitter.split_text(text.strip())
    return [
        ChunkIn(
            chunk_id=f"{doc_id}_{i}",
            doc_id=doc_id,
            chunk_index=i,
            text=c,
            doc_meta=doc_meta,
        )
        for i, c in enumerate(chunks)
        if c.strip()
    ]
