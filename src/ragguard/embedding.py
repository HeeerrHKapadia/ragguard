"""Text to vectors, via a local ONNX model.

Runs entirely on the machine — no API key, no per-call cost, and CI can
execute the real pipeline instead of a mock. The model is downloaded once
and cached.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from ragguard.config import settings


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    """Load the model once per process.

    Loading takes several seconds and the object is stateless afterwards, so
    a script that embedded per-call would spend most of its life re-reading
    the same weights off disk.
    """
    return TextEmbedding(model_name=settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus text for storage."""
    if not texts:
        return []
    return [vec.tolist() for vec in get_model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single search query.

    Separate from embed_documents because asymmetric models want a prefix on
    one side or the other. bge-small-en-v1.5 does not require it for
    retrieval, so this is currently identical — but keeping the call sites
    distinct means switching to a model that does need it is a one-line
    change here rather than a hunt through the codebase.
    """
    return next(iter(get_model().query_embed([text]))).tolist()
