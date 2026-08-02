"""Hybrid retrieval: lexical and dense, fused.

Dense embeddings match on meaning and are blind to exact strings. Ask about
a specific policy name, a fiscal year, a ticket prefix or a person's surname
and the nearest-neighbour search returns things that are *about* the same
topic while missing the document that actually contains the term.

Lexical search has the opposite failure: it matches the words that were
typed and nothing else, so "time off" never finds a page titled "vacation
policy".

Neither is a superset of the other, which is why fusing them beats either
alone on essentially every published benchmark.

A note on honesty in naming: the lexical half here is Postgres full-text
search ranked by ts_rank_cd, not BM25. They serve the same purpose but the
formulas differ — ts_rank_cd weights term density and proximity, while BM25
models term saturation and document length. Calling this BM25 because it is
the more familiar name would be a small lie that makes the results harder to
reproduce.
"""

from __future__ import annotations

import re

from ragguard.access import Principal
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc
from ragguard.retrieval.dense import _BaseDense
from ragguard.retrieval.filters import visibility_sql
from ragguard.retrieval.fusion import reciprocal_rank_fusion

WORD_RE = re.compile(r"[a-z0-9]+")

# How deep each ranker goes before fusion. Both lists need enough depth for
# a document that one ranker ranks poorly to still be rescued by the other —
# fusing two top-10 lists mostly just reranks what they already agreed on.
POOL = 60


def to_websearch_or(query: str) -> str:
    """Turn a query into an OR-joined websearch expression.

    Postgres treats space as AND in every tsquery constructor, so a five-word
    query would demand all five terms appear in the same chunk. That is a
    precision setting masquerading as a default, and it returns nothing for
    most real questions. Joining with OR and letting ts_rank_cd sort by how
    many terms matched is the behaviour people expect from search.

    websearch_to_tsquery is used rather than to_tsquery because it parses
    user-facing syntax and cannot be broken by punctuation.
    """
    terms = WORD_RE.findall(query.lower())
    return " OR ".join(terms)


class HybridRetriever(_BaseDense):
    """Dense and lexical retrieval over permitted rows, fused with RRF.

    Both channels apply the permission filter inside their own query, so
    neither ever considers a row the user may not see. Phase 2 established
    that filtering during retrieval rather than after it is worth several
    points of recall for low-privilege users; doing it in one channel and
    not the other would reintroduce exactly that loss.
    """

    name = "hybrid-rrf"

    def __init__(self, conn, oversample: int = 6, weights: tuple[float, float] = (1.0, 1.0)):
        super().__init__(conn, oversample)
        self.weights = weights

    def _dense_ranking(self, query: str, principal: Principal) -> list[str]:
        where, params = visibility_sql(principal)
        params["vec"] = self._embed(query)
        params["limit"] = POOL

        cur = self.conn.cursor()
        cur.execute(
            f"""SELECT d.source_uri
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                  JOIN tenants   t ON t.id = d.tenant_id
                 WHERE {where}
              ORDER BY c.embedding <=> %(vec)s
                 LIMIT %(limit)s""",
            params,
        )
        return dedupe([row[0] for row in cur.fetchall()])

    def _lexical_ranking(self, query: str, principal: Principal) -> list[str]:
        expression = to_websearch_or(query)
        if not expression:
            return []

        where, params = visibility_sql(principal)
        params["q"] = expression
        params["limit"] = POOL

        cur = self.conn.cursor()
        cur.execute(
            f"""SELECT d.source_uri
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                  JOIN tenants   t ON t.id = d.tenant_id
                 WHERE {where}
                   AND c.tsv @@ websearch_to_tsquery('english', %(q)s)
              ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(q)s)) DESC
                 LIMIT %(limit)s""",
            params,
        )
        return dedupe([row[0] for row in cur.fetchall()])

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        dense = self._dense_ranking(case.query, principal)
        lexical = self._lexical_ranking(case.query, principal)

        fused = reciprocal_rank_fusion([dense, lexical], weights=list(self.weights))[:k]
        if not fused:
            return []

        return self._hydrate([uri for uri, _ in fused], dict(fused))

    def _hydrate(self, uris: list[str], scores: dict[str, float]) -> list[RetrievedDoc]:
        """Attach tenant, section and tier to the fused URIs.

        The scorer re-resolves all of this against the corpus anyway, but a
        retriever that returns bare strings is useless outside the harness.
        """
        cur = self.conn.cursor()
        cur.execute(
            """SELECT d.source_uri, t.slug, d.section, d.sensitivity
                 FROM documents d JOIN tenants t ON t.id = d.tenant_id
                WHERE d.source_uri = ANY(%s)""",
            (uris,),
        )
        meta = {row[0]: row[1:] for row in cur.fetchall()}

        out: list[RetrievedDoc] = []
        for uri in uris:
            if uri in meta:
                tenant, section, tier = meta[uri]
                out.append(
                    RetrievedDoc(uri=uri, tenant=tenant, section=section,
                                 tier=tier, score=scores.get(uri, 0.0))
                )
        return out


def dedupe(items: list[str]) -> list[str]:
    """First occurrence wins, order preserved.

    Chunks are ranked but documents are returned, and a long document
    contributes many chunks. Without collapsing, one document could occupy
    most of a ranking and starve the fusion of alternatives.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
