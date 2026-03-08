"""Chroma implementation of IVectorStoreRepository. DB access only."""
from src.config import get_settings
from src.infrastructure.chroma_client import get_chroma_client
from src.schemas.chunk import ChunkIn, ChunkOut


class ChromaRepository:
    """Vector store using Chroma. No business logic – queries only."""

    def __init__(self, collection_name: str | None = None) -> None:
        self._client = get_chroma_client()
        name = collection_name or get_settings().chroma_collection
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"description": "Legal document chunks"},
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add_chunks(self, chunks: list[ChunkIn], embeddings: list[list[float]]) -> None:
        ids = [c.chunk_id for c in chunks]
        metadatas = []
        for c in chunks:
            meta: dict[str, str | int] = {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            if c.doc_meta:
                for k, v in c.doc_meta.items():
                    meta[k] = str(v) if v is not None else ""
            metadatas.append(meta)
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ChunkOut]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        ids = result["ids"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        # Chroma cosine: returns cosine distance (1 - similarity). Score = max(0, 1 - distance).
        out: list[ChunkOut] = []
        for i, (id_, meta, dist) in enumerate(zip(ids, metadatas, distances)):
            d = float(dist) if dist is not None else 1.0
            score = max(0.0, 1.0 - d)
            out.append(
                ChunkOut(
                    chunk_id=id_,
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    text=meta.get("text", ""),
                    score=min(1.0, score),
                    title=meta.get("title") or None,
                    citation=meta.get("citation") or None,
                    state=meta.get("state") or None,
                    issuer=meta.get("issuer") or None,
                    court=meta.get("court") or None,
                    date_filed=meta.get("date_filed") or None,
                    disposition=meta.get("disposition") or None,
                )
            )
        return out

    def get_embeddings_for_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Return embedding vectors for the given chunk ids. Missing ids are omitted."""
        if not ids:
            return {}
        result = self._collection.get(ids=ids, include=["embeddings"])
        out: dict[str, list[float]] = {}
        result_ids = result.get("ids") or []
        embs = result.get("embeddings")
        if embs is None or len(result_ids) == 0:
            return out
        for i, id_ in enumerate(result_ids):
            if i >= len(embs):
                break
            vec = embs[i]
            if vec is None:
                continue
            out[id_] = list(vec)
        return out
