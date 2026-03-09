"""Use case: search by query. Business logic and orchestration only."""
from collections import defaultdict
from typing import TYPE_CHECKING

from src.config import get_settings
from src.repositories import ChromaRepository, EmbeddingRepository
from src.schemas.chunk import ChunkOut
from src.schemas.query import SearchQuery
from src.schemas.search_result import RankedCase, SearchResult

if TYPE_CHECKING:
    from src.infrastructure.reranker_client import RerankerClient


class RetrievalService:
    """Orchestrates: query -> embed -> search [-> rerank] -> validate -> aggregate. No HTTP, no direct DB."""

    def __init__(
        self,
        vector_store: ChromaRepository | None = None,
        embedding_repo: EmbeddingRepository | None = None,
        reranker: "RerankerClient | None" = None,
    ) -> None:
        self._store = vector_store or ChromaRepository()
        self._embedding = embedding_repo or EmbeddingRepository()
        settings = get_settings()
        if reranker is not None:
            self._reranker = reranker
        elif settings.reranker_enabled:
            from src.infrastructure.reranker_client import get_reranker_client
            self._reranker = get_reranker_client()
        else:
            self._reranker = None
        # Debug: confirm reranker config at init
        print(f"[RERANKER INIT] reranker_enabled={settings.reranker_enabled}, _reranker={'injected' if reranker is not None else 'created' if self._reranker else 'None'}")

    def search(self, request: SearchQuery) -> SearchResult:
        """Return ranked cases: aggregate chunks by doc_id, score = mean of chunk scores."""
        settings = get_settings()
        top_k = request.top_k or settings.top_k
        min_words = settings.min_chunk_words
        min_chars = settings.min_chunk_chars

        if not request.query.strip():
            return SearchResult(query=request.query, cases=[])

        query_embedding = self._embedding.embed([request.query])[0]

        if self._reranker is not None:
            retrieve_k = settings.reranker_candidates
            chunks = self._store.search(query_embedding, top_k=retrieve_k)
            qpreview = request.query[:50] + "..." if len(request.query) > 50 else request.query
            print(f"[RERANKER] query='{qpreview}' ({len(request.query)} chars), chunks={len(chunks)}")
            before_scores = [round(c.score, 4) for c in chunks[:5]]
            chunks = self._reranker.rerank(request.query, chunks)
            after_scores = [round(c.score, 4) for c in chunks[:5]]
            print(f"[RERANKER] Before rerank (first 5): {before_scores}")
            print(f"[RERANKER] After rerank (first 5):  {after_scores}")
            chunks = chunks[: top_k * 2]
        else:
            print(f"[RERANKER] SKIP (reranker is None) – using Chroma scores only")
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
