"""Guarded generation: turn permitted retrieval into a cited answer.

The whole project's premise is that enforcement must happen *before* content
reaches the model. This package keeps that promise on the generation side:

  - `context.retrieve_context` fetches chunk text through the same permission
    filter the retriever uses, so the ONLY text a generator can ever see is
    text the persona is cleared for.
  - `Generator` implementations turn those permitted snippets into an answer
    whose every factual claim carries a citation to one of them.
  - The default `ExtractiveGenerator` needs no LLM and no API key: it composes
    the answer directly from permitted snippets, which makes it injection-immune
    by construction and a fixed anchor for the answer-level leak benchmark.

An optional LLM backend plugs in behind the same interface (see `llm.py`) and
is measured against the identical leak/faithfulness bar.
"""

from ragguard.generation.base import (
    Answer,
    Citation,
    Claim,
    Generator,
    Snippet,
)
from ragguard.generation.context import retrieve_context
from ragguard.generation.extractive import ExtractiveGenerator

__all__ = [
    "Answer",
    "Citation",
    "Claim",
    "Generator",
    "Snippet",
    "retrieve_context",
    "ExtractiveGenerator",
]
