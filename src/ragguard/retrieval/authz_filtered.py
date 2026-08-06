"""Retrieval where the authorization service decides, not the database.

The usual advice for putting an external authorization service in front of
search is: retrieve a generous candidate set from the index, then call the
service to check each result. That is post-filtering, and Phase 2 measured
what it costs — the least privileged persona lost 14.3 points of recall, and
closing the gap needed 50x oversampling.

The measurement in scripts/authz_benchmark.py points the other way.
ListObjects returns everything a user can reach in a single call, 220x
faster than checking each document, so the allowed set can be fetched
*before* the search and handed to the index as a filter. The authorization
service stays authoritative and retrieval stays pre-filtered.

**Where this stops working.** ListObjects returns a list, so its cost grows
with what the user can see. At 300 documents per tenant that is 1.9ms and an
easy win. At a million it is neither, and the architecture has to change —
materialise the allowed set into the index and accept staleness, or shard
the query. Worth stating plainly, because the pattern that is obviously
right here is not right at every size.
"""

from __future__ import annotations

import psycopg
from pgvector import Vector

from ragguard.access import Principal
from ragguard.authz.model import TENANT_SEP, safe_id
from ragguard.embedding import embed_query
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc

CHUNK_OVERSAMPLE = 6


def uri_from_object(obj: str) -> str:
    """Recover a document URI from an OpenFGA object id.

    Ids are `document:<tenant>--<sanitised uri>`, and sanitising is lossy —
    `gitlab://values.md` becomes `gitlab_//values.md`. Rather than trying to
    invert that, callers map back through a lookup built at load time. This
    returns the sanitised form for use as that lookup's key.
    """
    return obj.split(":", 1)[1].split(TENANT_SEP, 1)[1]


class AuthzPreFilterRetriever:
    """Ask the authorization service what is visible, then search only that.

    `allowed_uris` is supplied per principal rather than fetched inside
    retrieve(), because the OpenFGA client is async and the retrieval
    interface is not. In a real service this is one call at request start,
    cached for the life of the request.

    The allow-set is applied with ``JOIN unnest(...)`` rather than
    ``= ANY(list)``. Large array binds force the planner into sequential
    checks; a set join scales with the intersection instead.
    """

    name = "dense-authz"

    def __init__(self, conn: psycopg.Connection, allowed: dict[str, set[str]]) -> None:
        self.conn = conn
        self.allowed = allowed

    def _embed(self, text: str) -> Vector:
        return Vector(embed_query(text))

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        permitted = self.allowed.get(principal.email)
        if not permitted:
            return []

        cur = self.conn.cursor()
        cur.execute(
            """SELECT d.source_uri, t.slug, d.section, d.sensitivity,
                      c.embedding <=> %(vec)s AS distance,
                      d.title
                 FROM chunks c
                 JOIN documents d ON d.id = c.document_id
                 JOIN tenants   t ON t.id = d.tenant_id
                 JOIN unnest(%(allowed)s::text[]) AS allowed(uri)
                   ON allowed.uri = d.source_uri
             ORDER BY distance
                LIMIT %(limit)s""",
            {
                "vec": self._embed(case.query),
                "allowed": list(permitted),
                "limit": k * CHUNK_OVERSAMPLE,
            },
        )

        best: dict[str, RetrievedDoc] = {}
        for uri, tenant, section, tier, distance, title in cur.fetchall():
            if uri not in best:
                best[uri] = RetrievedDoc(
                    uri=uri, tenant=tenant, section=section,
                    tier=tier, score=float(distance), title=title or "",
                )
                if len(best) >= k:
                    break
        return list(best.values())


def build_allowed_map(
    principals: dict[str, Principal],
    objects_by_user: dict[str, list[str]],
    corpus_uris: set[str],
) -> dict[str, set[str]]:
    """Translate OpenFGA object ids back into document URIs.

    Sanitising is lossy, so the mapping is rebuilt from the corpus rather
    than inverted: each real URI is sanitised once and indexed by the result.
    """
    by_sanitised = {safe_id(uri): uri for uri in corpus_uris}

    out: dict[str, set[str]] = {}
    for email in principals:
        uris = set()
        for obj in objects_by_user.get(email, []):
            real = by_sanitised.get(uri_from_object(obj))
            if real is not None:
                uris.add(real)
        out[email] = uris
    return out
