"""Index CourtListener opinions into Chroma and optionally export. Supports resume via checkpoint."""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import get_settings
from src.infrastructure.courtlistener_client import stream_courtlistener_opinions
from src.infrastructure.embedding_client import embed as embed_fn
from src.repositories import ChromaRepository, EmbeddingRepository
from src.utils.chunking import split_into_chunks
from src.utils.embedding_sanity import run_embedding_sanity_check

CHECKPOINT_FILENAME = "courtlistener_checkpoint.json"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "doc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index CourtListener opinions; optional export. Resumes automatically from last checkpoint.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        metavar="N",
        help="Max opinions to index (default: no limit). Use 5 for a quick test.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/courtlistener_first_cases",
        help="Output folder for exported text and vector files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="(Deprecated) Resume from last checkpoint. Resuming is now the default when a checkpoint exists.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoint file (default: output-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting CourtListener indexer (download + embed + export)...", flush=True)
    settings = get_settings()

    if not run_embedding_sanity_check(embed_fn):
        print("Stopping. Fix the embedding model before indexing.", flush=True)
        sys.exit(1)
    output_root = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir or args.output_dir)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILENAME
    texts_dir = output_root / "texts"
    vectors_dir = output_root / "vectors"

    resume_from_url: str | None = None
    resume_total_docs = 0
    resume_total_chunks = 0

    # Try to load checkpoint if it exists – this is now the default resume behavior.
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            resume_from_url = data.get("next_url") or None
            resume_total_docs = int(data.get("indexed_docs") or 0)
            resume_total_chunks = int(data.get("indexed_chunks") or 0)
            print(f"Resuming from checkpoint: {checkpoint_path}")
            if resume_from_url:
                print("  next_url:", resume_from_url[:80] + "..." if len(resume_from_url) > 80 else resume_from_url)
            print(f"  totals so far: docs={resume_total_docs}, chunks={resume_total_chunks}")
        except Exception as e:
            print(f"Checkpoint read failed: {e}. Starting from scratch (could not read checkpoint: {e}).")
            resume_from_url = None
            resume_total_docs = 0
            resume_total_chunks = 0

    # לעולם לא מוחקים נתונים קיימים – הסקריפט רק ממשיך ומוסיף.
    texts_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    store = ChromaRepository(collection_name=settings.chroma_collection)
    embedding = EmbeddingRepository()

    session_chunks = 0
    session_docs = 0
    total_chunks = resume_total_chunks
    total_docs = resume_total_docs
    max_docs = args.max_docs  # None = no limit

    def on_page_done(next_url: str | None) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps({"next_url": next_url, "indexed_docs": total_docs, "indexed_chunks": total_chunks}, indent=2),
            encoding="utf-8",
        )

    for i, row in enumerate(
        stream_courtlistener_opinions(
            max_rows=max_docs,
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

        doc_meta: dict[str, str] = {}
        for key in ("id", "title", "citation", "docket_number", "court", "date_filed", "disposition"):
            if key in row and row[key] is not None:
                doc_meta[key] = str(row[key])
        if "id" not in doc_meta:
            doc_meta["id"] = doc_id

        chunks = split_into_chunks(doc_id, text, doc_meta=doc_meta if doc_meta else None)
        if not chunks:
            continue

        chunk_texts = [c.text for c in chunks]
        vectors = embedding.embed(chunk_texts)
        store.add_chunks(chunks, vectors)

        safe_doc = _safe_name(doc_id)
        texts_dir.joinpath(f"{safe_doc}.txt").write_text(text, encoding="utf-8")

        vector_payload = {
            "doc_id": doc_id,
            "meta": doc_meta,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "vector_dim": len(vec),
                    "vector": vec,
                    "text_preview": chunk.text[:200],
                }
                for chunk, vec in zip(chunks, vectors)
            ],
        }
        vectors_dir.joinpath(f"{safe_doc}.json").write_text(
            json.dumps(vector_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        session_chunks += len(chunks)
        session_docs += 1
        total_chunks += len(chunks)
        total_docs += 1
        limit_str = str(max_docs) if max_docs is not None else "∞"
        print(
            f"[{total_docs}/{limit_str}] indexed doc={doc_id} chunks={len(chunks)}"
        )

    print("\nDone.")
    print(f"Session indexed docs: {session_docs}")
    print(f"Session indexed chunks: {session_chunks}")
    print(f"Total indexed docs: {total_docs}")
    print(f"Total indexed chunks: {total_chunks}")
    print(f"Exported texts: {texts_dir}")
    print(f"Exported vectors: {vectors_dir}")
    if checkpoint_path.exists():
        print(f"Checkpoint saved: {checkpoint_path} (use --resume to continue)")


if __name__ == "__main__":
    main()
