from src.api.routes.analyze import router as analyze_router
from src.api.routes.cases import router as cases_router
from src.api.routes.index import router as index_router
from src.api.routes.search import router as search_router

__all__ = ["analyze_router", "cases_router", "index_router", "search_router"]
