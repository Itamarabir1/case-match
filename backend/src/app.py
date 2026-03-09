"""App bootstrap – wire config, routes, run server."""
import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    analyze_router,
    cases_router,
    index_router,
    search_router,
)
from src.api.setup import register_exception_handler, register_static_and_root
from src.config import get_settings

_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models and clients at startup so first request is fast."""
    from src.infrastructure.chroma_client import get_chroma_client
    from src.infrastructure.embedding_client import get_embedding_client
    from src.infrastructure.reranker_client import get_reranker_client

    s = get_settings()
    print("[STARTUP] Loading embedding model...")
    get_embedding_client()
    if s.reranker_enabled:
        print("[STARTUP] Loading reranker model...")
        get_reranker_client()
    print("[STARTUP] Connecting to Chroma...")
    get_chroma_client()
    print("[STARTUP] All models ready!")
    yield


def create_app() -> FastAPI:
    backend_root = Path(__file__).resolve().parent.parent
    app = FastAPI(
        title="Law Retrieval API",
        description="Semantic search over legal cases",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handler(app, _CORS_ORIGINS)

    app.include_router(search_router)
    app.include_router(index_router)
    app.include_router(analyze_router)
    app.include_router(cases_router)

    if os.getenv("API_ONLY", "").strip() != "1":
        register_static_and_root(app, backend_root)

    return app


app = create_app()


def run_server() -> None:
    """Run uvicorn. Used by backend/main.py when no subcommand or 'serve'."""
    import uvicorn

    s = get_settings()
    uvicorn.run("src.app:app", host="0.0.0.0", port=s.port, reload=(s.env == "development"))
