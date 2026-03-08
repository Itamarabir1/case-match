"""Embedding model client – sentence-transformers. No business logic."""
from sentence_transformers import SentenceTransformer, models

from src.config import get_settings

# Plain BERT (MLM) models that need explicit mean pooling + normalization for good embeddings.
_MODELS_NEED_POOLING = ("law-ai/InLegalBERT",)

_model: SentenceTransformer | None = None


def get_embedding_client() -> SentenceTransformer:
    """Return singleton SentenceTransformer. Loads on first call."""
    global _model
    if _model is None:
        settings = get_settings()
        name = settings.embedding_model
        if name in _MODELS_NEED_POOLING:
            transformer = models.Transformer(name, max_seq_length=512)
            pooling = models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode="mean",
            )
            normalize = models.Normalize()
            _model = SentenceTransformer(modules=[transformer, pooling, normalize])
        else:
            _model = SentenceTransformer(name)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts. Returns list of vectors."""
    client = get_embedding_client()
    return client.encode(texts, convert_to_numpy=True).tolist()
