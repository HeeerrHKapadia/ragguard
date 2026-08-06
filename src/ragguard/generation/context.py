"""Fetch the permitted context a generator is allowed to see.

This is the single choke point that makes generation leak-safe: it runs the
*same* permission policy the retriever uses (`visibility_sql_on_chunks`), so
the only chunk text that can ever leave this function is text the principal is
cleared for. A generator downstream cannot widen that set — it can only ever
narrow it.
"""

from __future__ import annotations

import psycopg
from pgvector import Vector

from ragguard.access import Principal
from ragguard.embedding import embed_query
from ragguard.generation.base import Snippet
from ragguard.retrieval.filters import visibility_sql_on_chunks

# Ask for several chunks per document then collapse, so one long document
# cannot crowd every other source out of the context window.
CHUNK_OVERSAMPLE = 6


def retrieve_context(
    conn: psycopg.Connection,
    query: str,
    principal: Principal,
    k: int = 5,
) -> list[Snippet]:
    """Return up to `k` permitted snippets, best chunk per document.

    The visibility filter runs inside the SQL, so forbidden rows are never
    even considered — identical guarantee to `PreFilterRetriever`, extended to
    carry the chunk text the generator needs.
    """
    where, params = visibility_sql_on_chunks(principal)
    params["vec"] = Vector(embed_query(query))
    params["limit"] = k * CHUNK_OVERSAMPLE

    cur = conn.cursor()
    cur.execute(
        f"""SELECT d.source_uri, d.title, t.slug, c.section, c.sensitivity,
                   c.text, c.embedding <=> %(vec)s AS distance
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
              JOIN tenants   t ON t.id = c.tenant_id
             WHERE {where}
          ORDER BY distance
             LIMIT %(limit)s""",
        params,
    )

    best: dict[str, Snippet] = {}
    for uri, title, tenant, section, tier, text, distance in cur.fetchall():
        if uri not in best:
            best[uri] = Snippet(
                uri=uri,
                title=title or "",
                tenant=tenant,
                section=section,
                tier=tier,
                text=text,
                score=float(distance),
            )
            if len(best) >= k:
                break
    return list(best.values())
