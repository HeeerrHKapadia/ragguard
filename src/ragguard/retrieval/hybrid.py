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

    Dense and lexical rankings run in a single SQL statement so the request
    path pays one round-trip instead of three (dense + lexical + hydrate).
    """

    name = "hybrid-rrf"

    def __init__(self, conn, oversample: int = 6, weights: tuple[float, float] = (1.0, 1.0)):
        super().__init__(conn, oversample)
        self.weights = weights

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        dense, lexical = self._rankings(case.query, principal)
        fused = reciprocal_rank_fusion([dense, lexical], weights=list(self.weights))[:k]
        if not fused:
            return []
        return self._hydrate([uri for uri, _ in fused], dict(fused))

    def _rankings(self, query: str, principal: Principal) -> tuple[list[str], list[str]]:
        """Return (dense_uris, lexical_uris) from one database round-trip."""
        where, params = visibility_sql(principal)
        expression = to_websearch_or(query)
        params["vec"] = self._embed(query)
        params["q"] = expression
        params["limit"] = POOL

        # Lexical half is skipped when the query has no searchable terms;
        # the CTE still runs but the tsquery match yields nothing.
        # Subqueries keep ORDER BY … LIMIT under the ANN / FTS plans. A bare
        # window over the filtered set would score every permitted chunk.
        # Ranks are assigned *after* the limit so the planner can stop early.
        cur = self.conn.cursor()
        cur.execute(
            f"""
            WITH
            q AS (
                SELECT CASE
                         WHEN %(q)s = '' THEN NULL
                         ELSE websearch_to_tsquery('english', %(q)s)
                       END AS tsq
            ),
            dense AS (
                SELECT uri, row_number() OVER (ORDER BY distance) AS rnk
                  FROM (
                    SELECT d.source_uri AS uri,
                           c.embedding <=> %(vec)s AS distance
                      FROM chunks c
                      JOIN documents d ON d.id = c.document_id
                      JOIN tenants   t ON t.id = d.tenant_id
                     WHERE {where}
                  ORDER BY distance
                     LIMIT %(limit)s
                  ) denserows
            ),
            lexical AS (
                SELECT uri, row_number() OVER (ORDER BY rank DESC) AS rnk
                  FROM (
                    SELECT d.source_uri AS uri,
                           ts_rank_cd(c.tsv, q.tsq) AS rank
                      FROM chunks c
                      JOIN documents d ON d.id = c.document_id
                      JOIN tenants   t ON t.id = d.tenant_id
                      CROSS JOIN q
                     WHERE {where}
                       AND q.tsq IS NOT NULL
                       AND c.tsv @@ q.tsq
                  ORDER BY rank DESC
                     LIMIT %(limit)s
                  ) lexrows
            )
            SELECT channel, uri FROM (
                SELECT 'dense'::text AS channel, uri, rnk FROM dense
                UNION ALL
                SELECT 'lexical', uri, rnk FROM lexical
            ) ranked
            ORDER BY channel, rnk
            """,
            params,
        )

        dense: list[str] = []
        lexical: list[str] = []
        for channel, uri in cur.fetchall():
            if channel == "dense":
                dense.append(uri)
            else:
                lexical.append(uri)
        return dedupe(dense), dedupe(lexical)

    def _hydrate(self, uris: list[str], scores: dict[str, float]) -> list[RetrievedDoc]:
        """Attach tenant, section, tier and title to the fused URIs."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT d.source_uri, t.slug, d.section, d.sensitivity, d.title
                 FROM documents d JOIN tenants t ON t.id = d.tenant_id
                WHERE d.source_uri = ANY(%s)""",
            (uris,),
        )
        meta = {row[0]: row[1:] for row in cur.fetchall()}

        out: list[RetrievedDoc] = []
        for uri in uris:
            if uri in meta:
                tenant, section, tier, title = meta[uri]
                out.append(
                    RetrievedDoc(
                        uri=uri, tenant=tenant, section=section,
                        tier=tier, score=scores.get(uri, 0.0), title=title or "",
                    )
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
