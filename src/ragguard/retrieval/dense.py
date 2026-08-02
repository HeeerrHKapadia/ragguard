"""Dense retrieval, in three degrees of authorization.

  DenseRetriever       no enforcement at all               (Phase 1 baseline)
  PostFilterRetriever  retrieve, then drop what is forbidden
  PreFilterRetriever   never consider forbidden rows

The last two implement the same policy and both reach a zero leak rate. The
difference between them is the entire point of Phase 2: *when* the filter
runs decides whether low-privilege users get a usable product.
"""

from __future__ import annotations

import psycopg
from pgvector import Vector

from ragguard.access import Principal, can_read
from ragguard.embedding import embed_query
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc
from ragguard.retrieval.filters import visibility_sql

# Chunks are retrieved but relevance is judged per document, and several
# chunks of one long document can crowd out every other document. Ask for
# more chunks than the document budget, then collapse them.
CHUNK_OVERSAMPLE = 6

SELECT_COLS = """SELECT d.source_uri, t.slug, d.section, d.sensitivity,
                        c.embedding <=> %(vec)s AS distance"""

FROM_JOIN = """FROM chunks c
               JOIN documents d ON d.id = c.document_id
               JOIN tenants   t ON t.id = d.tenant_id"""


class _BaseDense:
    """Shared query embedding and chunk-to-document collapsing."""

    name = "dense"

    def __init__(self, conn: psycopg.Connection, oversample: int = CHUNK_OVERSAMPLE) -> None:
        self.conn = conn
        self.oversample = oversample
        # Each query is asked by three or four personas, so without this the
        # same text is embedded three or four times. 884 cases collapse to
        # 220 distinct queries.
        self._cache: dict[str, Vector] = {}

    def _embed(self, text: str) -> Vector:
        cached = self._cache.get(text)
        if cached is None:
            cached = Vector(embed_query(text))
            self._cache[text] = cached
        return cached

    @staticmethod
    def _collapse(rows, k: int) -> list[RetrievedDoc]:
        """Keep the best-scoring chunk per document, up to k documents."""
        best: dict[str, RetrievedDoc] = {}
        for uri, tenant, section, tier, distance in rows:
            if uri not in best:
                best[uri] = RetrievedDoc(
                    uri=uri, tenant=tenant, section=section,
                    tier=tier, score=float(distance),
                )
                if len(best) >= k:
                    break
        return list(best.values())


class DenseRetriever(_BaseDense):
    """Phase 1: nearest neighbours, no authorization anywhere.

    A faithful implementation of a large share of shipped RAG systems, where
    "access control" lives in a system prompt asking the model not to use
    documents the user should not see. By then the content is already in the
    context window, so the instruction is a request, not a control.
    """

    name = "dense-naive"

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        cur = self.conn.cursor()
        cur.execute(
            f"{SELECT_COLS} {FROM_JOIN} ORDER BY distance LIMIT %(limit)s",
            {"vec": self._embed(case.query), "limit": k * self.oversample},
        )
        return self._collapse(cur.fetchall(), k)


class PostFilterRetriever(_BaseDense):
    """Retrieve first, then discard what the user may not see.

    The intuitive fix, and the one that quietly fails. The database still
    selects the globally nearest chunks with no idea who is asking, so the
    candidate pool a low-privilege user gets back is mostly material they
    cannot be shown. Dropping those rows afterwards leaves them with a
    handful of results while an executive gets a full page — from the same
    query, with the same code, and with a leak rate of zero.

    Nothing in a conventional RAG scorecard moves when this happens.
    """

    name = "dense-postfilter"

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        cur = self.conn.cursor()
        cur.execute(
            f"{SELECT_COLS} {FROM_JOIN} ORDER BY distance LIMIT %(limit)s",
            {"vec": self._embed(case.query), "limit": k * self.oversample},
        )
        permitted = [
            row for row in cur.fetchall()
            if can_read(principal, row[1], row[2], row[3])
        ]
        return self._collapse(permitted, k)


class PreFilterRetriever(_BaseDense):
    """Never consider a row the user may not see.

    The filter runs inside the query, so every one of the candidate slots is
    spent on something the user can actually be shown. Identical policy to
    the post-filter version and an identical zero leak rate — the difference
    is that a new hire's budget is no longer consumed by rows that were
    always going to be discarded.
    """

    name = "dense-prefilter"

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        where, params = visibility_sql(principal)
        params["vec"] = self._embed(case.query)
        params["limit"] = k * self.oversample

        cur = self.conn.cursor()
        cur.execute(
            f"{SELECT_COLS} {FROM_JOIN} WHERE {where} ORDER BY distance LIMIT %(limit)s",
            params,
        )
        return self._collapse(cur.fetchall(), k)
