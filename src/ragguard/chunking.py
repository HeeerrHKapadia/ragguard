"""Split documents into chunks.

Phase 1 uses fixed-size character windows with overlap, which is the naive
strategy on purpose. It ignores everything a markdown document tells you
about its own structure: headings, section boundaries, table rows, code
fences. A chunk routinely begins mid-sentence under one heading and ends
under another.

That is the baseline being measured, not a recommendation. Phase 2 replaces
it with structure-aware splitting and the eval reports what that was worth.
Building the good version first would leave nothing to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

# ~4 characters per token is the usual English rule of thumb, so 1600
# characters lands near 400 tokens. The embedding model truncates at 512
# tokens, and silent truncation is the worst kind of data loss — content
# past the cutoff is indexed as though it were never written. Staying under
# the limit costs nothing and removes the failure mode.
CHUNK_CHARS = 1600
OVERLAP_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str


def chunk_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Split into overlapping fixed-size windows.

    The overlap exists because a hard cut can land in the middle of the one
    sentence that answers a question, leaving neither chunk able to answer
    it. Repeating the tail of each window at the head of the next means any
    span shorter than the overlap survives intact in at least one chunk.
    """
    text = text.strip()
    if not text:
        return []
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    stride = chunk_chars - overlap_chars

    while start < len(text):
        window = text[start:start + chunk_chars].strip()
        if window:
            chunks.append(Chunk(ordinal=ordinal, text=window))
            ordinal += 1
        if start + chunk_chars >= len(text):
            break
        start += stride

    return chunks


def estimate_tokens(text: str) -> int:
    """Rough token count. Good enough for reporting, not for budgeting."""
    return len(text) // 4
