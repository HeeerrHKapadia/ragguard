"""Wire the pieces together: corpus + principals + cases -> a scored report."""

from __future__ import annotations

from ragguard.access import Principal
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import Report, RetrievedDoc, score_case
from ragguard.eval.retriever import Retriever


def load_corpus_index(cur) -> dict[str, RetrievedDoc]:
    """Every document's true tenant, section, and tier, keyed by URI.

    This is the authority the scorer trusts. A retriever may return whatever
    it likes; how that result is judged comes from here.
    """
    cur.execute(
        """SELECT d.source_uri, t.slug, d.section, d.sensitivity
             FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
    )
    return {
        uri: RetrievedDoc(uri=uri, tenant=tenant, section=section, tier=tier)
        for uri, tenant, section, tier in cur.fetchall()
    }


def run(
    retriever: Retriever,
    cases: list[GoldenCase],
    principals: dict[str, Principal],
    corpus: dict[str, RetrievedDoc],
    k: int = 10,
) -> Report:
    report = Report()
    for case in cases:
        principal = principals.get(case.persona)
        if principal is None:
            continue
        retrieved = retriever.retrieve(case, principal, k)
        report.results.append(score_case(case, principal, retrieved, corpus))
    return report
