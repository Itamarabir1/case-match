"""CLI: run a search query and print results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.schemas.query import SearchQuery
from src.services.retrieval_service import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for similar legal cases by query text.")
    parser.add_argument("query", nargs="*", help="Query text (or leave empty to type interactively).")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="Return the K most similar cases (default from settings).")
    args = parser.parse_args()

    if not args.query:
        query_text = input("Enter legal problem (query): ").strip()
    else:
        query_text = " ".join(args.query).strip()
    if not query_text:
        print("Empty query.")
        sys.exit(1)
    request = SearchQuery(query=query_text, top_k=args.top_k)
    service = RetrievalService()
    result = service.search(request)
    print(f"Query: {result.query}")
    print(f"Found {len(result.cases)} cases.")
    for i, case in enumerate(result.cases, 1):
        pct = round(case.score * 100)
        print(f"\n{i}. doc_id={case.doc_id}  Score: {pct}% דמיון  chunks={case.chunk_count}")
        for snip in case.snippets[:2]:
            print(f"   {snip[:100]}...")


if __name__ == "__main__":
    main()
