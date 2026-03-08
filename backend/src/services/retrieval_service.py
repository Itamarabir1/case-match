"""Use case: search by query. Business logic and orchestration only."""
from collections import defaultdict

from src.config import get_settings
from src.repositories import ChromaRepository, EmbeddingRepository
from src.schemas.chunk import ChunkOut
from src.schemas.query import SearchQuery
from src.schemas.search_result import RankedCase, SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates: query -> embed -> search -> validate -> aggregate. No HTTP, no direct DB."""

    def __init__(
        self,
        vector_store: ChromaRepository | None = None,
        embedding_repo: EmbeddingRepository | None = None,
    ) -> None:
        self._store = vector_store or ChromaRepository()
        self._embedding = embedding_repo or EmbeddingRepository()

    def search(self, request: SearchQuery) -> SearchResult:
        """Return ranked cases: aggregate chunks by doc_id, score = mean of chunk scores."""
        settings = get_settings()
        top_k = request.top_k or settings.top_k
        min_words = settings.min_chunk_words
        min_chars = settings.min_chunk_chars

        if not request.query.strip():
            return SearchResult(query=request.query, cases=[])

        query_embedding = self._embedding.embed([request.query])[0]
        chunks = self._store.search(query_embedding, top_k=top_k * 2)

        # Validate: drop too short
        valid = [
            c
            for c in chunks
            if c.text and len(c.text.strip()) >= min_chars and len(c.text.strip().split()) >= min_words
        ]

        # Aggregate by doc_id: mean score, collect snippets, keep first chunk for doc metadata
        by_doc: dict[str, list[float]] = defaultdict(list)
        snippets_by_doc: dict[str, list[str]] = defaultdict(list)
        first_chunk_by_doc: dict[str, ChunkOut] = {}
        for c in valid:
            by_doc[c.doc_id].append(c.score)
            if len(snippets_by_doc[c.doc_id]) < 3:
                snippets_by_doc[c.doc_id].append(c.text[:200] + ("..." if len(c.text) > 200 else ""))
            if c.doc_id not in first_chunk_by_doc:
                first_chunk_by_doc[c.doc_id] = c

        cases = []
        for doc_id, scores in by_doc.items():
            first = first_chunk_by_doc.get(doc_id)
            cases.append(
                RankedCase(
                    doc_id=doc_id,
                    score=sum(scores) / len(scores),
                    chunk_count=len(scores),
                    snippets=snippets_by_doc.get(doc_id, [])[:5],
                    title=first.title if first else None,
                    citation=first.citation if first else None,
                    state=first.state if first else None,
                    issuer=first.issuer if first else None,
                    court=first.court if first else None,
                    date_filed=first.date_filed if first else None,
                    disposition=first.disposition if first else None,
                )
            )
        cases.sort(key=lambda x: x.score, reverse=True)
        cases = cases[:top_k]

        best_chunk_id = valid[0].chunk_id if valid else None
        best_chunk_score = float(valid[0].score) if valid else None

        return SearchResult(
            query=request.query,
            cases=cases,
            query_embedding=query_embedding,
            best_chunk_id=best_chunk_id,
            best_chunk_score=best_chunk_score,
        )
