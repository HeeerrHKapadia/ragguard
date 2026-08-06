"""Split documents into chunks.

Fixed-size windows are the measured Phase 1 baseline. Structure-aware
splitting is available via ``strategy="markdown"`` for the next re-index:
it prefers heading boundaries so a chunk rarely begins under one section
and ends under another, which cuts useless overlap and improves lexical
density without changing the embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ~4 characters per token is the usual English rule of thumb, so 1600
# characters lands near 400 tokens. The embedding model truncates at 512
# tokens, and silent truncation is the worst kind of data loss — content
# past the cutoff is indexed as though it were never written. Staying under
# the limit costs nothing and removes the failure mode.
CHUNK_CHARS = 1600
OVERLAP_CHARS = 200

_HEADING_RE = re.compile(r"(?m)^(#{1,6}\s+\S.*)$")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str


def chunk_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    strategy: str = "fixed",
) -> list[Chunk]:
    """Split text into overlapping windows.

    ``strategy="fixed"`` is the naive baseline (Phase 1).
    ``strategy="markdown"`` packs heading-aware sections into windows first,
    then falls back to fixed windows for oversized sections.
    """
    if strategy == "markdown":
        return chunk_markdown(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    return chunk_fixed(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)


def chunk_fixed(
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


def chunk_markdown(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Heading-aware packing with fixed-window fallback for long sections."""
    text = text.strip()
    if not text:
        return []

    parts = _HEADING_RE.split(text)
    sections: list[str] = []
    buf = parts[0].strip() if parts else ""
    i = 1
    while i < len(parts):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        section = f"{heading}\n\n{body}".strip() if body else heading
        if buf:
            sections.append(buf)
            buf = ""
        sections.append(section)
        i += 2
    if buf:
        sections.append(buf)
    if not sections:
        return chunk_fixed(text, chunk_chars, overlap_chars)

    packed: list[str] = []
    current = ""
    for section in sections:
        if len(section) > chunk_chars:
            if current:
                packed.append(current)
                current = ""
            for piece in chunk_fixed(section, chunk_chars, overlap_chars):
                packed.append(piece.text)
            continue
        candidate = f"{current}\n\n{section}".strip() if current else section
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = section
    if current:
        packed.append(current)

    return [Chunk(ordinal=i, text=t) for i, t in enumerate(packed) if t]


def estimate_tokens(text: str) -> int:
    """Rough token count. Good enough for reporting, not for budgeting."""
    return len(text) // 4
