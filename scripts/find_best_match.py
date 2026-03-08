"""CLI: find which indexed case (and vector) best matches a legal question."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.schemas.query import SearchQuery
from src.services.retrieval_service import RetrievalService

DEFAULT_QUERY = "Can a neighbor's Wi-Fi password be enough for probable cause?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the case and vector in the index that best match a legal question.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        default=None,
        help=f"Legal question (default: \"{DEFAULT_QUERY}\").",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to show (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_text = " ".join(args.query).strip() if args.query else DEFAULT_QUERY
    if not query_text:
        print("Empty query.")
        sys.exit(1)

    request = SearchQuery(query=query_text, top_k=args.top_k)
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

    print("Query:", result.query)
    print()
    print("התיקים הכי מתאימים (לפי דמיון וקטורי):")
    print("=" * 60)
    for i, case in enumerate(result.cases, 1):
        label = "הכי מתאים" if i == 1 else f"#{i}"
        pct = round(case.score * 100)
        print(f"\n{label} – doc_id={case.doc_id}  Score: {pct}% דמיון  chunks={case.chunk_count}")
        if case.title:
            print(f"   title: {case.title}")
        if case.citation:
            print(f"   citation: {case.citation}")
        if case.issuer:
            print(f"   issuer: {case.issuer}")
        if case.state:
            print(f"   state: {case.state}")
        for j, snip in enumerate(case.snippets[:2], 1):
            print(f"   snippet {j}: {snip[:200]}...")
    print()
    print("התיק שהכי מתאים:", result.cases[0].doc_id)


if __name__ == "__main__":
    main()
