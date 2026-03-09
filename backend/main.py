"""Entry point. From project root: python backend/main.py (server) or python backend/main.py build-index (download data).
From backend dir: python main.py or python main.py build-index."""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_backend = Path(__file__).resolve().parent
sys.path.insert(0, str(_backend))
os.chdir(_root)
from dotenv import load_dotenv
load_dotenv(_root / ".env")
load_dotenv(_backend / ".env")


def _run_build_index() -> None:
    """CLI: download and index CourtListener data; resume from checkpoint; print progress."""
    from src.services.index_service import build_index, CHECKPOINT_FILENAME, DEFAULT_OUTPUT_DIR

    max_docs = None
    output_dir = DEFAULT_OUTPUT_DIR
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--max-docs" and i + 1 < len(args):
            max_docs = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
            continue
        i += 1

    root = Path.cwd()
    checkpoint_path = root / output_dir / CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        import json
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            docs = int(data.get("indexed_docs") or 0)
            chunks = int(data.get("indexed_chunks") or 0)
            print(f"Resuming from checkpoint: {docs} docs, {chunks} chunks so far.")
        except Exception:
            print("Resuming from checkpoint (counts unknown).")
    else:
        print("No checkpoint found – starting from the beginning.")

    if max_docs is not None:
        print(f"Limit: {max_docs} documents (will stop when total reaches this).")
    else:
        print("No limit – will continue until all pages are fetched (Ctrl+C to stop and save progress).")
    print("---")

    def on_progress(docs: int, total_chunks: int, chunks_this_doc: int) -> None:
        print(f"  Indexed: {docs} docs, {chunks_this_doc} chunks (this case)")

    try:
        build_index(max_docs=max_docs, output_dir=output_dir, on_progress=on_progress)
        print("\nDone.")
    except KeyboardInterrupt:
        print("\nStopped by user. Progress saved in checkpoint; run again to continue.")


def _run_rollback() -> None:
    """CLI: rollback to a doc count (remove docs indexed after that number)."""
    from src.services.index_service import rollback_to_doc_count, DEFAULT_OUTPUT_DIR

    target = 47861
    output_dir = DEFAULT_OUTPUT_DIR
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--to-docs" and i + 1 < len(args):
            target = int(args[i + 1])
            i += 2
            continue
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
            continue
        i += 1
    result = rollback_to_doc_count(target_docs=target, output_dir=output_dir)
    print(result.get("message", result))
    print("Indexed docs:", result.get("indexed_docs"), "| Chunks:", result.get("indexed_chunks"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build-index":
        _run_build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "rollback":
        _run_rollback()
    else:
        from src.app import run_server
        run_server()
