"""The default generator: deterministic, no LLM, injection-immune.

It composes the answer directly from the permitted snippets — one cited claim
per source — so:

  - every claim is grounded in a permitted source by construction
    (faithfulness is always 100%);
  - it cannot be "instructed" by text inside a document, because it never
    interprets snippet text as commands — it only quotes it. That makes it a
    natural control in the prompt-injection red-team: an attack that steers an
    LLM into leaking has nothing to steer here.

An empty result is a neutral refusal that does not reveal whether any forbidden
documents exist (guarding against the A3 existence-disclosure risk).
"""

from __future__ import annotations

import re

from ragguard.access import Principal
from ragguard.generation.base import Answer, Citation, Claim, Generator, Snippet

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")

EXCERPT_MAX_CHARS = 240


def _excerpt(text: str, max_chars: int = EXCERPT_MAX_CHARS) -> str:
    """First sentence or two of a chunk, whitespace-normalized.

    Kept short and verbatim: the extractive generator's value is that it never
    paraphrases, so a claim always says exactly what its cited source says.
    """
    clean = _WHITESPACE.sub(" ", text).strip()
    if not clean:
        return ""
    sentences = _SENTENCE_END.split(clean)
    out = ""
    for sentence in sentences:
        candidate = (out + " " + sentence).strip() if out else sentence
        if len(candidate) > max_chars and out:
            break
        out = candidate
        if len(out) >= max_chars:
            break
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "…"
    return out


class ExtractiveGenerator(Generator):
    """Compose a cited answer purely from permitted snippets."""

    name = "extractive"

    def generate(
        self, query: str, principal: Principal, snippets: list[Snippet]
    ) -> Answer:
        if not snippets:
            return Answer(
                query=query,
                persona=principal.email,
                backend=self.name,
                preamble=(
                    "I don't have any material you're permitted to access that "
                    "answers this question."
                ),
            )

        citations: list[Citation] = []
        claims: list[Claim] = []
        for i, snip in enumerate(snippets, start=1):
            excerpt = _excerpt(snip.text)
            if not excerpt:
                continue
            citations.append(
                Citation(marker=i, uri=snip.uri, title=snip.title, tier=snip.tier)
            )
            claims.append(Claim(text=excerpt, citation=i))

        preamble = (
            f"Based on {len(claims)} source(s) you're permitted to access:"
            if claims
            else "I don't have any material you're permitted to access that "
            "answers this question."
        )
        return Answer(
            query=query,
            persona=principal.email,
            backend=self.name,
            claims=tuple(claims),
            citations=tuple(citations),
            preamble=preamble,
        )
