"""Index operations: build from CourtListener, reset, stats. Used by API and CLI script."""
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from src.config import get_settings
from src.infrastructure.courtlistener_client import stream_courtlistener_opinions
from src.infrastructure.embedding_client import embed as embed_fn
from src.repositories import ChromaRepository, EmbeddingRepository
from src.utils.chunking import split_into_chunks
from src.utils.embedding_sanity import run_embedding_sanity_check
from src.utils.text import safe_filename

CHECKPOINT_FILENAME = "courtlistener_checkpoint.json"
DEFAULT_OUTPUT_DIR = "exports/courtlistener_first_cases"


def _root() -> Path:
    """Project root when run via main.py (cwd); else backend root in Docker."""
    return Path.cwd()


def get_index_stats() -> dict:
    """Return index stats for GET /index/stats."""
    settings = get_settings()
    from src.infrastructure.chroma_client import get_chroma_client
    client = get_chroma_client()
    coll = client.get_or_create_collection(name=settings.chroma_collection)
    return {"collection": settings.chroma_collection, "total_chunks": coll.count()}


def reset_index(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Delete Chroma DB and exports. Returns summary."""
    settings = get_settings()
    root = _root()
    chroma_path = root / settings.chroma_path
    exports_path = root / output_dir
    removed = []
    for path, label in [(chroma_path, "Chroma DB"), (exports_path, "Exports")]:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(label)
    return {"removed": removed, "message": "Index and exports cleared. Use Build index to start over."}


def build_index(
    max_docs: int | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> None:
    """Build or resume index from CourtListener. Long-running – call from background task or CLI.
    Reads checkpoint from output_dir/courtlistener_checkpoint.json; continues from next_url and counts.
    Optional on_progress(docs, total_chunks, chunks_this_doc) called after each document."""
    settings = get_settings()
    root = _root()
    output_root = root / output_dir
    checkpoint_dir = output_root
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    texts_dir = output_root / "texts"
    vectors_dir = output_root / "vectors"

    if not run_embedding_sanity_check(embed_fn):
        raise RuntimeError("Embedding sanity check failed. Fix the model before indexing.")

    resume_from_url: str | None = None
    total_docs = 0
    total_chunks = 0
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            resume_from_url = data.get("next_url")
            total_docs = int(data.get("indexed_docs") or 0)
            total_chunks = int(data.get("indexed_chunks") or 0)
        except Exception:
            pass

    # When resuming with max_docs, only fetch (max_docs - total_docs) more
    fetch_limit: int | None = None
    if max_docs is not None:
        fetch_limit = max(0, max_docs - total_docs)
    if fetch_limit == 0:
        return  # already at or past max_docs

    texts_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store = ChromaRepository(collection_name=settings.chroma_collection)
    embedding = EmbeddingRepository()

    def on_page_done(next_url: str | None) -> None:
        checkpoint_path.write_text(
            json.dumps({"next_url": next_url, "indexed_docs": total_docs, "indexed_chunks": total_chunks}, indent=2),
            encoding="utf-8",
        )

    for i, row in enumerate(
        stream_courtlistener_opinions(
            max_rows=fetch_limit,
            resume_from_url=resume_from_url,
            on_page_done=on_page_done,
            fetch_title_from_cluster=True,
        ),
        start=1,
    ):
        doc_id = str(row.get("id", i))
        text = str(row.get("document", "")).strip()
        if not text:
            continue
        doc_meta = {k: str(row[k]) for k in ("id", "title", "citation", "docket_number", "court", "date_filed", "disposition") if k in row and row[k] is not None}
        if "id" not in doc_meta:
            doc_meta["id"] = doc_id
        chunks = split_into_chunks(doc_id, text, doc_meta=doc_meta or None)
        if not chunks:
            continue
        vectors = embedding.embed([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        texts_dir.joinpath(f"{safe_filename(doc_id)}.txt").write_text(text, encoding="utf-8")
        vector_payload = {
            "doc_id": doc_id,
            "meta": doc_meta,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "vector_dim": len(vec),
                    "vector": vec,
                    "text_preview": (c.text[:200] + "...") if len(c.text) > 200 else c.text,
                }
                for c, vec in zip(chunks, vectors)
            ],
        }
        vectors_dir.joinpath(f"{safe_filename(doc_id)}.json").write_text(
            json.dumps(vector_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_docs += 1
        total_chunks += len(chunks)
        if on_progress:
            on_progress(total_docs, total_chunks, len(chunks))


def rollback_to_doc_count(
    target_docs: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Remove all data (Chroma chunks + text files) for docs indexed after target_docs.
    If checkpoint already says target_docs, still fetches one page from next_url and removes those
    doc ids (handles case where docs were added but checkpoint was not updated)."""
    import urllib.request

    settings = get_settings()
    root = _root()
    output_root = root / output_dir
    checkpoint_path = output_root / CHECKPOINT_FILENAME
    texts_dir = output_root / "texts"
    vectors_dir = output_root / "vectors"
    token = (settings.courtlistener_api_token or "").strip()
    if not token:
        raise ValueError("COURTLISTENER_API_TOKEN required for rollback (to fetch page and get doc ids).")

    if not checkpoint_path.exists():
        return {"removed_docs": 0, "message": "No checkpoint found."}

    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    current_docs = int(data.get("indexed_docs") or 0)
    next_url = data.get("next_url")
    if not next_url:
        return {"removed_docs": 0, "message": "No next_url in checkpoint."}

    store = ChromaRepository(collection_name=settings.chroma_collection)
    removed_docs = 0
    while True:
        req = urllib.request.Request(
            next_url,
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            page_data = json.loads(resp.read().decode("utf-8"))
        results = page_data.get("results") or []
        doc_ids_to_remove = [str(op.get("id")) for op in results if op.get("id") is not None]
        if not doc_ids_to_remove:
            next_url = page_data.get("next")
            if not next_url:
                break
            continue
        for doc_id in doc_ids_to_remove:
            store.delete_chunks_by_doc_id(doc_id)
            safe = safe_filename(doc_id)
            txt_path = texts_dir / f"{safe}.txt"
            if txt_path.exists():
                txt_path.unlink()
            vec_path = vectors_dir / f"{safe}.json"
            if vec_path.exists():
                vec_path.unlink()
            removed_docs += 1
            current_docs = max(0, current_docs - 1)
        next_url = page_data.get("next")
        if current_docs <= target_docs:
            break
        if not next_url:
            break

    chunk_count = store._collection.count()
    final_docs = target_docs if removed_docs > 0 else current_docs
    checkpoint_path.write_text(
        json.dumps(
            {"next_url": next_url, "indexed_docs": final_docs, "indexed_chunks": chunk_count},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "removed_docs": removed_docs,
        "indexed_docs": final_docs,
        "indexed_chunks": chunk_count,
        "message": f"Rolled back to {final_docs} docs. Removed {removed_docs} docs from Chroma and texts.",
    }
