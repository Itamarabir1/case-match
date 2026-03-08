"""Decision Support: statistics on similar cases – plaintiff vs defendant wins, success probability."""
import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "backend"))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")
load_dotenv(_project_root / "backend" / ".env")

from src.schemas.query import SearchQuery
from src.services.retrieval_service import RetrievalService

# Number of similar cases to use for stats.
DEFAULT_TOP_K = 5

# Mapping disposition text to winner (extend for your jurisdiction).
# US appellate: "Reversed" often = appellant won; "Affirmed" = appellee won.
DISPOSITION_PLAINTIFF_WINS = ("reversed", "vacated", "remanded", "reversal")
DISPOSITION_DEFENDANT_WINS = ("affirmed", "affirmance")


def _classify_winner(disposition: str | None) -> str | None:
    """Return 'plaintiff', 'defendant', or None if unknown."""
    if not disposition or not disposition.strip():
        return None
    d = disposition.strip().lower()
    for kw in DISPOSITION_PLAINTIFF_WINS:
        if kw in d:
            return "plaintiff"
    for kw in DISPOSITION_DEFENDANT_WINS:
        if kw in d:
            return "defendant"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision Support: plaintiff vs defendant win stats and success probability.",
    )
    parser.add_argument(
        "query_or_file",
        nargs="?",
        default=None,
        help="Query text or path to a .txt file with case description.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="K",
        help="Number of similar cases to consider.",
    )
    args = parser.parse_args()

    top_k = args.top_k

    if not args.query_or_file:
        query_text = input("Enter legal problem or path to case file: ").strip()
    else:
        p = Path(args.query_or_file)
        if p.exists() and p.is_file() and p.suffix.lower() in (".txt",):
            query_text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not query_text:
                print("File is empty.")
                sys.exit(1)
        else:
            query_text = args.query_or_file.strip()

    if not query_text:
        print("Empty query.")
        sys.exit(1)

    request = SearchQuery(query=query_text, top_k=top_k)
    service = RetrievalService()
    try:
        result = service.search(request)
    except Exception as exc:
        print(f"Search failed: {exc}")
        sys.exit(1)

    cases = result.cases
    if not cases:
        print("No similar cases found in the index.")
        sys.exit(0)

    plaintiff_wins = 0
    defendant_wins = 0
    unknown = 0
    for case in cases:
        winner = _classify_winner(case.disposition)
        if winner == "plaintiff":
            plaintiff_wins += 1
        elif winner == "defendant":
            defendant_wins += 1
        else:
            unknown += 1

    total_with_outcome = plaintiff_wins + defendant_wins
    if total_with_outcome > 0:
        pct_plaintiff = round(100 * plaintiff_wins / total_with_outcome)
    else:
        pct_plaintiff = 0

    print()
    print("=" * 60)
    print("Decision Support – סטטיסטיקה על תיקים דומים")
    print("=" * 60)
    print(f"מתוך {len(cases)} תיקים דומים:")
    print(f"  התובע ניצח:  {plaintiff_wins}")
    print(f"  הנתבע ניצח:  {defendant_wins}")
    if unknown:
        print(f"  לא ידוע:      {unknown}")
    print()
    if total_with_outcome > 0:
        print(f"הסתברות הצלחה לתובע: {pct_plaintiff}%")
    else:
        print("הסתברות הצלחה לתובע: — (אין תיקים עם disposition מזוהה)")
        print("טיפ: וודא שהאינדקס נבנה עם disposition מ-CourtListener.")
    print("=" * 60)


if __name__ == "__main__":
    main()
