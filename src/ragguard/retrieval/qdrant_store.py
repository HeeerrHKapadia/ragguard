"""Qdrant: the same vectors, with the permission filter inside the index.

Phase 2 found that pre-filtering in Postgres halves throughput — 492 queries
per second unfiltered against 243 with a selective `WHERE`. The cause is
structural: pgvector's HNSW index is built over every row, so a filter is
applied around the search rather than inside it, and the engine has to walk
further to find enough surviving candidates.

Qdrant indexes payload fields alongside the vectors and evaluates conditions
during graph traversal, which is claimed to make filtered search roughly as
fast as unfiltered regardless of how selective the filter is. This is the
fifth expression of the same access policy, so it gets the same treatment as
the other four: verified against the oracle, not trusted.
"""

from __future__ import annotations

import os

from qdrant_client import QdrantClient, models

from ragguard.access import TIER_RANK, Principal
from ragguard.config import settings

COLLECTION = "ragguard_chunks"
TIERS = ["public", "internal", "confidential", "restricted"]


def client(prefer_grpc: bool = True) -> QdrantClient:
    """Qdrant client, over gRPC by default.

    The comparison against pgvector is otherwise unfair: pgvector answers on
    an already-open local socket while Qdrant pays HTTP framing per request.
    At single-digit milliseconds that overhead is the same magnitude as the
    difference being measured, so it has to come out of the comparison rather
    than be reported as an index result.
    """
    host = os.getenv("QDRANT_HOST", "localhost")
    return QdrantClient(
        host=host,
        port=int(os.getenv("QDRANT_PORT", "6335")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6336")),
        prefer_grpc=prefer_grpc,
        timeout=60,
    )


def ensure_collection(qc: QdrantClient, dim: int = 0) -> None:
    """Create the collection with indexed payload fields.

    The payload indexes are the entire point. Without them Qdrant still
    filters correctly but has to check conditions row by row, which is the
    behaviour this benchmark exists to compare against rather than reproduce.
    """
    dim = dim or settings.embedding_dim
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)

    qc.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    for field in ("tenant", "section", "tier", "uri"):
        qc.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def visibility_filter(principal: Principal) -> models.Filter:
    """The access policy as a Qdrant filter.

    Structure mirrors can_read(): tenant is a hard requirement, then either
    the document's tier is within the principal's clearance, or the document
    sits in a section where one of their groups is elevated.

    `must` and `should` together mean: every `must` holds AND at least one
    `should` holds.
    """
    base_tiers = [
        tier for tier in TIERS
        if TIER_RANK[tier] <= TIER_RANK[principal.max_clearance]
    ]

    alternatives: list[models.Condition] = [
        models.FieldCondition(key="tier", match=models.MatchAny(any=base_tiers))
    ]

    for grant in principal.grants:
        if not grant.elevated_sections:
            continue
        reachable = [
            tier for tier in TIERS
            if TIER_RANK[tier] <= TIER_RANK[grant.clearance] + 1
        ]
        alternatives.append(
            models.Filter(must=[
                models.FieldCondition(
                    key="section",
                    match=models.MatchAny(any=list(grant.elevated_sections)),
                ),
                models.FieldCondition(key="tier", match=models.MatchAny(any=reachable)),
            ])
        )

    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant",
                match=models.MatchValue(value=principal.tenant_slug),
            )
        ],
        should=alternatives,
    )


def search(qc: QdrantClient, vector: list[float], principal: Principal,
           limit: int) -> list[tuple[str, str, str, str, float]]:
    """Filtered nearest-neighbour search. Returns (uri, tenant, section, tier, score)."""
    hits = qc.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=visibility_filter(principal),
        limit=limit,
        with_payload=True,
    ).points

    return [
        (h.payload["uri"], h.payload["tenant"], h.payload["section"],
         h.payload["tier"], float(h.score))
        for h in hits
    ]
