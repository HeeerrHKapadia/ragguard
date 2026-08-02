"""Cross-encoder reranking.

The embedding model is a *bi-encoder*: it turns the query into a vector and
each document into a vector separately, then compares them. That separation
is what makes search fast — every document can be embedded once, in advance,
and the query only has to be compared against stored vectors. It is also
what limits accuracy, because the model never sees the query and the
document together and cannot notice that a document mentions the right term
in the wrong sense.

A *cross-encoder* reads the query and one document as a single input and
scores the pair directly. Far more accurate, and far too slow to run against
a whole corpus: it cannot precompute anything, so scoring 4996 chunks means
4996 forward passes per query.

The standard resolution is to use both. The bi-encoder retrieves a shortlist
cheaply, then the cross-encoder rescoring only that shortlist. Accuracy of
the expensive model at close to the cost of the cheap one — for the
documents that made the shortlist. Anything the first stage missed stays
missed, which is why the shortlist has to be generously sized.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg
from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragguard.access import Principal
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc

# 80MB, trained on MS MARCO. Small enough to run on CPU inside CI, which
# matters more here than the last point of accuracy from a 1GB model.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# How many documents the first stage hands over. Too few and reranking can
# only shuffle what was already right; too many and the cost stops being
# amortised. 40 into 10 is a common production ratio.
SHORTLIST = 40


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL)


class RerankingRetriever:
    """Wrap any retriever and rescore its shortlist with a cross-encoder.

    Composes rather than subclasses, so the same reranker can sit on top of
    dense, hybrid, or later the graph retriever without duplicating it three
    times — and so the measured contribution is attributable to reranking
    alone rather than to a reimplementation of the stage beneath it.
    """

    def __init__(self, inner, conn: psycopg.Connection, shortlist: int = SHORTLIST) -> None:
        self.inner = inner
        self.conn = conn
        self.shortlist = shortlist
        self.name = f"{inner.name}+rerank"

    def _document_text(self, uris: list[str]) -> dict[str, str]:
        """Fetch representative text for scoring.

        The cross-encoder reads text, not vectors, so the documents have to
        come back from the database. Only the first few chunks are used:
        the model truncates at 512 tokens anyway, and feeding it a whole
        handbook page would mean paying for text it silently discards.
        """
        cur = self.conn.cursor()
        cur.execute(
            """SELECT d.source_uri,
                      string_agg(c.text, ' ' ORDER BY c.ordinal) AS body
                 FROM documents d
                 JOIN chunks c ON c.document_id = d.id
                WHERE d.source_uri = ANY(%s) AND c.ordinal < 2
             GROUP BY d.source_uri""",
            (uris,),
        )
        return {uri: body[:2000] for uri, body in cur.fetchall()}

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        candidates = self.inner.retrieve(case, principal, self.shortlist)
        if len(candidates) <= 1:
            return candidates[:k]

        texts = self._document_text([doc.uri for doc in candidates])
        scorable = [doc for doc in candidates if doc.uri in texts]
        if not scorable:
            return candidates[:k]

        scores = list(
            get_reranker().rerank(case.query, [texts[doc.uri] for doc in scorable])
        )

        ranked = sorted(zip(scorable, scores), key=lambda pair: -pair[1])
        return [
            RetrievedDoc(
                uri=doc.uri, tenant=doc.tenant, section=doc.section,
                tier=doc.tier, score=float(score),
            )
            for doc, score in ranked[:k]
        ]
