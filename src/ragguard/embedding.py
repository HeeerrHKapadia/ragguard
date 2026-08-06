"""Text to vectors, via a local ONNX model.

Runs entirely on the machine — no API key, no per-call cost, and CI can
execute the real pipeline instead of a mock. The model is downloaded once
and cached.
"""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

from fastembed import TextEmbedding

from ragguard.config import settings

# The eval harness asks the same 220 queries under many personas and many
# retrievers. Caching per retriever instance still re-embeds across them;
# a process-wide cache collapses that to one forward pass per distinct text.
_QUERY_CACHE: dict[str, list[float]] = {}
_QUERY_CACHE_LOCK = Lock()
_QUERY_CACHE_MAX = 2048


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
    cached = _QUERY_CACHE.get(text)
    if cached is not None:
        return cached

    vector = next(iter(get_model().query_embed([text]))).tolist()
    with _QUERY_CACHE_LOCK:
        if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
            # Drop an arbitrary oldest-ish entry; exact LRU is not worth the
            # complexity for a bounded demo/eval cache.
            _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)), None)
        _QUERY_CACHE[text] = vector
    return vector


def clear_query_cache() -> None:
    """Test helper."""
    with _QUERY_CACHE_LOCK:
        _QUERY_CACHE.clear()
