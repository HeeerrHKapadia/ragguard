"""Phase 1: dense-only retrieval with no authorization whatsoever.

This is the baseline the whole project improves on, and it is deliberately
built the way most first-pass RAG systems are built:

  * embed the query, take the nearest chunks, done
  * no tenant filter, no permission check, no reranking
  * "access control" would live in a system prompt telling the model not to
    use documents the user should not see

That last point is the one worth dwelling on. Prompt-level enforcement is
not enforcement — the forbidden content is already in the context window by
the time the instruction is read, so a leak is one prompt injection or one
unlucky sampling away. The eval measures retrieval, so it sees the leak that
a prompt instruction would merely be papering over.

Nothing here is a strawman. This is a faithful implementation of the
architecture in a large share of shipped RAG systems.
"""

from __future__ import annotations

import psycopg
from pgvector import Vector

from ragguard.access import Principal
from ragguard.embedding import embed_query
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc

# Chunks are retrieved, but relevance is judged per document. Several chunks
# of one long document can crowd out every other document, so ask for more
# chunks than the document budget and collapse them afterwards.
CHUNK_OVERSAMPLE = 6


class DenseRetriever:
    """Nearest-neighbour search over chunk embeddings."""

    name = "dense-naive"

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn
        # Every query is asked by three or four personas, so the same text
        # would otherwise be embedded three or four times. 884 cases collapse
        # to 220 distinct queries.
        self._query_cache: dict[str, Vector] = {}

    def _embed(self, text: str) -> Vector:
        cached = self._query_cache.get(text)
        if cached is None:
            cached = Vector(embed_query(text))
            self._query_cache[text] = cached
        return cached

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        vec = self._embed(case.query)

        # No WHERE clause. Not an oversight — the absence of `tenant_id = ...`
        # here is the entire finding this phase exists to produce.
        cur = self.conn.cursor()
        cur.execute(
            """SELECT d.source_uri, t.slug, d.section, d.sensitivity,
                      c.embedding <=> %s AS distance
                 FROM chunks c
                 JOIN documents d ON d.id = c.document_id
                 JOIN tenants   t ON t.id = d.tenant_id
             ORDER BY distance
                LIMIT %s""",
            (vec, k * CHUNK_OVERSAMPLE),
        )

        best: dict[str, RetrievedDoc] = {}
        for uri, tenant, section, tier, distance in cur.fetchall():
            if uri not in best:
                best[uri] = RetrievedDoc(
                    uri=uri, tenant=tenant, section=section,
                    tier=tier, score=float(distance),
                )
            if len(best) >= k:
                break

        return list(best.values())
