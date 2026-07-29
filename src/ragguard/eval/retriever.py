"""The retriever interface, plus three reference implementations.

An evaluation harness is itself software, and software that grades other
software is unusually easy to get wrong in a way nobody notices — a scorer
with an inverted comparison happily reports encouraging numbers forever.

The fix is to calibrate it against systems whose correct score is known in
advance. If the harness cannot report 100% leaks for a retriever that
deliberately ignores permissions, the harness is broken, and every number it
produces about a real retriever is worthless.

These three bracket the space:

    NullRetriever    returns nothing   -> no leaks, no recall
    LeakyRetriever   returns the world -> leaks everywhere, perfect recall
    OracleRetriever  returns exactly   -> no leaks, perfect recall
                     what is allowed
                     and relevant

The real Phase 1 retriever slots into the same Protocol and lands somewhere
between them.
"""

from __future__ import annotations

from typing import Protocol

from ragguard.access import Principal, can_read
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc


class Retriever(Protocol):
    """Anything that can answer a query on behalf of a principal.

    The principal is passed in rather than the retriever looking it up,
    because authorization must be an input to retrieval — not something
    applied afterwards to whatever came back.
    """

    name: str

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        ...


class NullRetriever:
    """Returns nothing. The trivially secure, entirely useless system.

    Worth keeping permanently: it is the reminder that leak rate alone is
    not a success criterion. This scores a perfect zero leaks.
    """

    name = "null"

    def __init__(self, corpus: dict[str, RetrievedDoc]) -> None:
        self.corpus = corpus

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        return []


class LeakyRetriever:
    """Perfect relevance, zero authorization — the Phase 1 baseline in spirit.

    Returns the relevant documents and then pads the remaining slots from
    anywhere in the corpus, including other tenants. That padding is the
    honest part: a system with no filtering does not politely stop at the
    documents that happen to be relevant, it returns whatever ranked highest
    across everything it indexed.

    (An earlier version returned only relevant documents and scored a mild
    32% leak rate, because it leaked solely when a relevant document was
    itself forbidden. It modelled a system that was accidentally safe most of
    the time, which is not the baseline worth measuring against.)
    """

    name = "leaky"

    def __init__(self, corpus: dict[str, RetrievedDoc]) -> None:
        self.corpus = corpus
        # Stable order so runs are reproducible.
        self._all = [corpus[uri] for uri in sorted(corpus)]

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        docs = [self.corpus[uri] for uri in case.relevant_uris if uri in self.corpus][:k]
        seen = {d.uri for d in docs}
        for doc in self._all:
            if len(docs) >= k:
                break
            if doc.uri not in seen:
                docs.append(doc)
        return docs


class OracleRetriever:
    """The unreachable ideal: everything allowed, nothing forbidden.

    No real retriever hits this, because it has been handed the answer key.
    It exists to prove the harness can recognise a perfect score, and to put
    a ceiling on the results table.
    """

    name = "oracle"

    def __init__(self, corpus: dict[str, RetrievedDoc]) -> None:
        self.corpus = corpus

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        allowed = [
            doc for uri in case.relevant_uris
            if (doc := self.corpus.get(uri))
            and can_read(principal, doc.tenant, doc.section, doc.tier)
        ]
        return allowed[:k]
