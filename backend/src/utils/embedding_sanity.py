"""Sanity check for embedding model: ensure different texts get different vectors."""
import math
from collections.abc import Callable

# Two completely different sentences – legal vs unrelated.
SANITY_TEXT_LEGAL = (
    "A landlord evicted a tenant without proper notice and kept the security deposit."
)
SANITY_TEXT_UNRELATED = "Mix flour, sugar and eggs to bake a chocolate cake."

# If cosine similarity between these is above this, the model likely does not discriminate.
SANITY_MAX_SIM = 0.9


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Handles normalized or unnormalized."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_embedding_sanity_check(embed_fn: Callable[[list[str]], list[list[float]]]) -> bool:
    """Run sanity check: embed two different sentences and check cosine similarity.

    Args:
        embed_fn: Callable that takes list[str] and returns list[list[float]].

    Returns:
        True if check passed (similarity <= SANITY_MAX_SIM), False otherwise.
    """
    vecs = embed_fn([SANITY_TEXT_LEGAL, SANITY_TEXT_UNRELATED])
    if len(vecs) != 2:
        print("Embedding sanity check failed: expected 2 vectors.")
        return False
    sim = _cosine_similarity(vecs[0], vecs[1])
    print(f"Embedding sanity check: cosine similarity (legal vs cake) = {sim:.4f}")
    if sim > SANITY_MAX_SIM:
        print(
            f"WARNING: similarity {sim:.4f} > {SANITY_MAX_SIM}. "
            "The model may not be discriminating between different inputs. "
            "Check pooling and normalization for plain BERT models."
        )
        return False
    print("Embedding sanity check passed.")
    return True
