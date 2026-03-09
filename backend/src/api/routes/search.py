"""Search API: POST /search – query text → ranked cases."""
from fastapi import APIRouter

from src.schemas.query import SearchQuery
from src.schemas.search_result import SearchResult
from src.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResult)
def search_endpoint(body: SearchQuery) -> SearchResult:
    """Legal problem query → ranked similar cases."""
    return RetrievalService().search(body)
