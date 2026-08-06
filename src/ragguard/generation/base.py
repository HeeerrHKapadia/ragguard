"""Data contracts for guarded generation.

An answer is not free text. It is a list of **claims**, each of which must
point at a **citation** — a permitted source snippet it was derived from. That
shape is deliberate: it makes "is every statement grounded in something this
persona may read?" a mechanical check rather than a judgement call, which is
exactly what turns leak-safety into a hard, testable property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragguard.access import Principal


@dataclass(frozen=True)
class Snippet:
    """A single permitted source chunk handed to a generator.

    By construction every Snippet has already passed the visibility filter in
    `context.retrieve_context`, so a generator physically cannot be given text
    the persona may not read. `tier`/`section`/`tenant` are carried only so the
    answer-level leak metric can re-verify against the corpus oracle.
    """

    uri: str
    title: str
    tenant: str
    section: str
    tier: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class Citation:
    """A numbered source reference an answer's claims point at."""

    marker: int          # the [n] shown in the rendered answer
    uri: str
    title: str
    tier: str


@dataclass(frozen=True)
class Claim:
    """One factual statement in an answer.

    `citation` is the marker of the source it was drawn from, or None for a
    statement the generator could not ground. An ungrounded claim is treated as
    a faithfulness failure by the checker — it is either a hallucination or, in
    the worst case, leaked content with no legitimate source.
    """

    text: str
    citation: int | None = None


@dataclass(frozen=True)
class Answer:
    """The full generated response, in a form that can be graded exactly."""

    query: str
    persona: str
    backend: str
    claims: tuple[Claim, ...] = ()
    citations: tuple[Citation, ...] = ()
    preamble: str = ""

    @property
    def text(self) -> str:
        """Render the answer as human-readable text with [n] markers."""
        lines: list[str] = []
        if self.preamble:
            lines.append(self.preamble)
        for claim in self.claims:
            marker = f" [{claim.citation}]" if claim.citation is not None else ""
            lines.append(f"- {claim.text}{marker}")
        if self.citations:
            lines.append("")
            lines.append("Sources:")
            for c in self.citations:
                lines.append(f"  [{c.marker}] {c.title} ({c.tier}) — {c.uri}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "persona": self.persona,
            "backend": self.backend,
            "preamble": self.preamble,
            "claims": [{"text": c.text, "citation": c.citation} for c in self.claims],
            "citations": [
                {"marker": c.marker, "uri": c.uri, "title": c.title, "tier": c.tier}
                for c in self.citations
            ],
            "text": self.text,
        }


class Generator(ABC):
    """Turns permitted snippets into a cited answer.

    Implementations receive only snippets that already passed the permission
    filter; they must never fetch their own context. The contract is: every
    Claim must cite a Snippet that was passed in, and the generator must not
    invent URIs or content outside the provided snippets.
    """

    name: str = "generator"

    @abstractmethod
    def generate(
        self, query: str, principal: Principal, snippets: list[Snippet]
    ) -> Answer:
        raise NotImplementedError


@dataclass
class GenerationResult:
    """Answer plus the snippets it was built from, for grading and tracing."""

    answer: Answer
    snippets: list[Snippet] = field(default_factory=list)
