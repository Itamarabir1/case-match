"""App bootstrap – wire config, routes, run server."""
import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import search_router
from src.config import get_settings
from src.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from src.services.rag_service import GroqUnavailableError, run_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config on startup. No heavy init (Chroma/embedding lazy)."""
    s = get_settings()
    print(f"[RERANKER] At app startup: reranker_enabled={s.reranker_enabled}")
    yield
    # shutdown if needed
    pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Law Retrieval API",
        description="Semantic search over legal cases",
        lifespan=lifespan,
    )
    app.include_router(search_router)

    # POST /analyze – RAG: retrieval + Groq analysis
    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze_endpoint(body: AnalyzeRequest) -> AnalyzeResponse:
        """Run retrieval and Groq LLM analysis on the given query."""
        try:
            cases, analysis, model = run_rag(query=body.query, top_k=body.top_k)
            return AnalyzeResponse(query=body.query, cases=cases, analysis=analysis, model=model)
        except GroqUnavailableError as e:
            raise HTTPException(status_code=503, detail=e.message)
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))

    # When API_ONLY=1 (e.g. backend container), skip static; frontend is served separately
    api_only = os.getenv("API_ONLY", "").strip() == "1"
    if not api_only:
        root_dir = Path(__file__).resolve().parent.parent
        frontend_dir = root_dir / "frontend"
        static_dir = root_dir / "static"
        if frontend_dir.exists():
            app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
        elif static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        @app.get("/")
        def root():
            return RedirectResponse(url="/static/index.html")
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=settings.port,
        reload=(settings.env == "development"),
    )
