"""Cross-encoder reranker for (query, chunk) scoring. Second stage after vector search."""
import math

from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.schemas.chunk import ChunkOut
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Temperature > 1 makes scores less extreme; calibration maps sigmoid output to target ranges
RERANKER_TEMPERATURE = 2.5  # try 2.0 or 3.0 for more spread
RERANKER_SCALE = 0.65       # so best ~ 0.65 + 0.2 = 0.85
RERANKER_SHIFT = 0.2        # so unrelated ~ 0.2


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calibrate(raw_score: float) -> float:
    """Convert raw logit to [0,1] via sigmoid with temperature, then calibrate to ~75-85% for good match, ~15-30% for unrelated."""
    s = _sigmoid(raw_score / RERANKER_TEMPERATURE)
    return min(1.0, max(0.0, RERANKER_SCALE * s + RERANKER_SHIFT))


class RerankerClient:
    """Rerank chunks by relevance to the query using a cross-encoder. Scores normalized to [0, 1]."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[ChunkOut]) -> list[ChunkOut]:
        """Score (query, chunk) pairs with the cross-encoder, normalize to [0,1], return sorted by score desc."""
        qpreview = (query[:50] + "...") if len(query) > 50 else query
        print(f"[RERANKER] rerank() called: query='{qpreview}', chunks={len(chunks)}")
        if not chunks:
            return []

        before = [round(c.score, 4) for c in chunks[:5]]
        pairs = [(query, c.text) for c in chunks]
        model = self._get_model()
        raw_scores = model.predict(pairs)

        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]
        raw_scores = [float(s) for s in raw_scores]

        # Log raw scores from CrossEncoder before any normalization
        print(f"[RERANKER] raw scores (first 5, before normalization): {[round(s, 4) for s in raw_scores[:5]]}")

        # Sigmoid + temperature + calibration instead of min-max (avoids 100% and too-high scores)
        norm_scores = [_calibrate(s) for s in raw_scores]

        out = [
            c.model_copy(update={"score": ns})
            for c, ns in zip(chunks, norm_scores)
        ]
        out.sort(key=lambda x: x.score, reverse=True)
        after = [round(c.score, 4) for c in out[:5]]
        print(f"[RERANKER] Before rerank (first 5 chunk scores): {before}")
        print(f"[RERANKER] After rerank (first 5, calibrated): {after}")
        return out
