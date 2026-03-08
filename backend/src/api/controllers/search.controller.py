"""Search controller: parse request, call pipeline, return response. No business logic."""
from src.pipeline.query_pipeline import run_query_pipeline
from src.schemas.query import SearchQuery
from src.schemas.search_result import SearchResult


def search(request: SearchQuery) -> SearchResult:
    """Handle search request: retrieval only."""
    return run_query_pipeline(request.query, top_k=request.top_k)
