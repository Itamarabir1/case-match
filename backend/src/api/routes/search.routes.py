"""Search routes: define endpoints and bind to controller."""
from fastapi import APIRouter

from src.api.controllers.search.controller import search
from src.schemas.query import SearchQuery
from src.schemas.search_result import SearchResult

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResult)
def search_endpoint(body: SearchQuery) -> SearchResult:
    """POST /search – legal problem query, returns ranked cases."""
    return search(body)
