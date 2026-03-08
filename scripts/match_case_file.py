"""CLI: match a legal case text file to the most similar indexed cases. Shows query vector and best-matching chunk vector."""
import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "backend"))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")
load_dotenv(_project_root / "backend" / ".env")

from src.repositories.chroma_repository import ChromaRepository
from src.schemas.query import SearchQuery
from src.services.retrieval_service import RetrievalService

VECTOR_PREVIEW_LEN = 10


DEFAULT_CASE_FILE = "examples/new_case.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the most similar indexed legal cases for a case text file.",
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help=f"Path to a .txt file (default: {DEFAULT_CASE_FILE}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of ranked cases to return (default from settings).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file_path) if args.file_path else _project_root / DEFAULT_CASE_FILE

    if not path.is_absolute():
        path = (_project_root / path).resolve()
    if not path.exists() or not path.is_file():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        print(f"Error: file is empty: {path}")
        sys.exit(1)

    request = SearchQuery(query=content, top_k=args.top_k)
    service = RetrievalService()
    try:
        result = service.search(request)
    except Exception as exc:
        print(f"Search failed: {exc}")
        print("Tip: ensure the Chroma index is healthy and not partially corrupted.")
        sys.exit(1)

    if not result.cases:
        print("No similar cases found in the current index.")
        sys.exit(0)

    print(f"Input file: {path}")
    print(f"Query length: {len(content)} chars")
    print(f"Found {len(result.cases)} similar case(s).")

    if result.query_embedding:
        qv = result.query_embedding
        preview = qv[:VECTOR_PREVIEW_LEN]
        print("\n" + "=" * 72)
        print("וקטור השאילתה (אחרי המרה):")
        print(f"  אורך: {len(qv)} מספרים")
        print(f"  תצוגה (ראשוני {VECTOR_PREVIEW_LEN}): {preview}")
        print("=" * 72)

    if result.best_chunk_id and result.best_chunk_score is not None:
        store = ChromaRepository()
        vecs = store.get_embeddings_for_ids([result.best_chunk_id])
        if vecs and result.best_chunk_id in vecs:
            v = vecs[result.best_chunk_id]
            preview = v[:VECTOR_PREVIEW_LEN]
            pct = round(result.best_chunk_score * 100)
            print("\n" + "=" * 72)
            print("הווקטור הכי מתאים באינדקס:")
            print(f"  chunk_id: {result.best_chunk_id}")
            print(f"  ציון דמיון: {pct}%")
            print(f"  אורך וקטור: {len(v)} מספרים")
            print(f"  תצוגה (ראשוני {VECTOR_PREVIEW_LEN}): {preview}")
            print("=" * 72)

    print()
    for i, case in enumerate(result.cases, 1):
        print("\n" + "=" * 72)
        pct = round(case.score * 100)
        print(f"{i}. doc_id={case.doc_id}  Score: {pct}% דמיון  chunks={case.chunk_count}")
        if case.title:
            print(f"   title: {case.title}")
        if case.citation:
            print(f"   citation: {case.citation}")
        if case.state:
            print(f"   state: {case.state}")
        if case.issuer:
            print(f"   issuer: {case.issuer}")
        for snip in case.snippets[:2]:
            print(f"   snippet: {snip[:180]}...")


if __name__ == "__main__":
    main()
