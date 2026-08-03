"""Per-request traces, kept in memory.

Langfuse was the plan and it needs an account, so this is a local stand-in:
structured spans, a ring buffer, and an endpoint to read them. The point is
the same either way — being able to answer "why did this request return
these documents, and where did the time go" without adding print statements
to a running service.

What a trace deliberately does *not* record is how many documents were
withheld. Phase 8 identified result-count disclosure as an open risk, and a
trace that surfaced the hidden count would turn a latent signal into an API
endpoint that serves it.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

# Bounded so a long-running process cannot fill memory with its own history.
MAX_TRACES = 200


@dataclass
class Span:
    name: str
    ms: float


@dataclass
class Trace:
    persona: str
    query: str
    spans: list[Span] = field(default_factory=list)
    results: int = 0
    total_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "persona": self.persona,
            "query": self.query,
            "results": self.results,
            "total_ms": round(self.total_ms, 1),
            "spans": [{"name": s.name, "ms": round(s.ms, 1)} for s in self.spans],
        }


_traces: deque[Trace] = deque(maxlen=MAX_TRACES)


@contextmanager
def trace(persona: str, query: str) -> Iterator[Trace]:
    record = Trace(persona=persona, query=query)
    started = time.perf_counter()
    try:
        yield record
    finally:
        record.total_ms = (time.perf_counter() - started) * 1000
        _traces.appendleft(record)


@contextmanager
def span(record: Trace, name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        record.spans.append(Span(name=name, ms=(time.perf_counter() - started) * 1000))


def recent(limit: int = 20) -> list[dict]:
    return [t.as_dict() for t in list(_traces)[:limit]]
