"""App setup: exception handler (CORS-aware) and static files + root redirect."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


def register_exception_handler(app: FastAPI, cors_origins: tuple[str, ...]) -> None:
    """Register 500 handler that adds CORS Allow-Origin from request."""

    @app.exception_handler(Exception)
    def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        origin = request.headers.get("origin", "")
        allow_origin = origin if origin in cors_origins else cors_origins[0]
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
            headers={"Access-Control-Allow-Origin": allow_origin},
        )


def register_static_and_root(app: FastAPI, backend_root: Path) -> None:
    """Mount /static and GET / -> /static/index.html when frontend/ or static/ exists."""
    frontend_dir = backend_root / "frontend"
    static_dir = backend_root / "static"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    elif static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    if frontend_dir.exists() or static_dir.exists():

        @app.get("/")
        def root():
            return RedirectResponse(url="/static/index.html")
