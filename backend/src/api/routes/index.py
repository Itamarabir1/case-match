"""Index API: build, reset, stats. For use from the web UI."""
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.services.index_service import build_index, get_index_stats, reset_index

router = APIRouter(prefix="/index", tags=["index"])


@router.get("/stats")
def index_stats():
    """GET /index/stats – return collection name and total chunks."""
    return get_index_stats()


@router.post("/reset")
def index_reset():
    """POST /index/reset – delete Chroma DB and exports. Next build starts from scratch."""
    return reset_index()


@router.post("/rebuild")
def index_rebuild(background_tasks: BackgroundTasks, max_docs: int | None = None):
    """POST /index/rebuild – start index build in background. Optional max_docs for a limited run."""
    from src.infrastructure.embedding_client import embed as embed_fn
    from src.utils.embedding_sanity import run_embedding_sanity_check
    if not run_embedding_sanity_check(embed_fn):
        raise HTTPException(status_code=500, detail="Embedding sanity check failed. Fix the model before indexing.")
    background_tasks.add_task(build_index, max_docs=max_docs)
    return {"status": "started", "message": "Index build started in background. This may take several minutes."}
