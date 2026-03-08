"""Reset all indexed data and checkpoint so the next run of rebuild_first_courtlistener_cases starts from scratch.

Deletes:
- Chroma DB (chroma_path from settings, default: chroma_db)
- Exported CourtListener data: texts, vectors, and courtlistener_checkpoint.json (default: exports/courtlistener_first_cases)

After running this script, run rebuild_first_courtlistener_cases.py to index from the beginning.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import get_settings

DEFAULT_EXPORTS_DIR = "exports/courtlistener_first_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete Chroma DB, exported vectors/texts, and checkpoint. Next index run will start from scratch.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_EXPORTS_DIR,
        help="Exports directory to remove (default: same as rebuild script).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be deleted, do not delete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    root = Path.cwd()
    chroma_path = root / settings.chroma_path
    exports_path = root / args.output_dir

    to_remove: list[tuple[Path, str]] = []
    if chroma_path.exists():
        to_remove.append((chroma_path, "Chroma DB"))
    if exports_path.exists():
        to_remove.append((exports_path, "Exports (texts, vectors, checkpoint)"))

    if not to_remove:
        print("Nothing to remove: Chroma DB and exports directory are already absent.")
        return

    if args.dry_run:
        print("Dry run – would remove:")
        for path, label in to_remove:
            print(f"  - {label}: {path}")
        return

    for path, label in to_remove:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Removed {label}: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}", file=sys.stderr)
            sys.exit(1)

    print("Done. Next run of rebuild_first_courtlistener_cases.py will start from the beginning.")


if __name__ == "__main__":
    main()
