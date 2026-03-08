"""Show what's in the Chroma index: full document text + vector + doc metadata per chunk."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.config import get_settings
from src.infrastructure.chroma_client import get_chroma_client


def main() -> None:
    settings = get_settings()
    client = get_chroma_client()
    coll = client.get_or_create_collection(name=settings.chroma_collection)
    n = coll.count()
    print(f"Collection: {settings.chroma_collection}")
    print(f"Total chunks: {n}\n")
    if n == 0:
        print("(Empty – run rebuild_first_courtlistener_cases.py first.)")
        return

    # Get all: ids, full metadata (text), and embeddings (vector)
    data = coll.get(include=["metadatas", "embeddings"])
    ids = data["ids"]
    metadatas = data["metadatas"] or []
    embeddings = data["embeddings"] or []

    # Doc-level metadata keys we may have stored (e.g. from CourtListener)
    meta_keys = ["title", "citation", "docket_number", "state", "issuer", "timestamp", "id"]

    for i, (chunk_id, meta, vec) in enumerate(zip(ids, metadatas, embeddings)):
        meta = meta or {}
        doc_id = meta.get("doc_id", "?")
        text = meta.get("text", "")

        print("=" * 60)
        print(f"תיק (doc_id): {doc_id}")
        print(f"chunk_id: {chunk_id}")
        doc_meta = {k: meta.get(k) for k in meta_keys if meta.get(k) is not None and meta.get(k) != ""}
        if doc_meta:
            print("מטא-דאטה (תיק):")
            for k, v in doc_meta.items():
                print(f"  {k}: {v}")
        print("-" * 60)
        print("טקסט התיק (המסמך):")
        print(text)
        print("-" * 60)
        print("הווקטור (embedding) – רשימת המספרים:")
        if vec:
            print(vec)
            print(f"(אורך: {len(vec)} מספרים)")
        else:
            print("(לא נשמר)")
        print()

    print("=" * 60)
    print("סיום – כל התיקים והווקטורים למעלה.")


if __name__ == "__main__":
    main()
