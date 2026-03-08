"""Pipeline: query -> embed -> search -> aggregate."""
from src.schemas.query import SearchQuery
from src.schemas.search_result import SearchResult
from src.services.retrieval_service import RetrievalService


def run_query_pipeline(query_text: str, top_k: int | None = None) -> SearchResult:
    """Run full query pipeline. Returns ranked cases."""
    request = SearchQuery(query=query_text, top_k=top_k)
    service = RetrievalService()
    return service.search(request)
