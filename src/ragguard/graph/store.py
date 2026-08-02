"""Neo4j access, and the graph's shape.

The schema is small on purpose:

    (:Document {uri, tenant, tier, section, title})
    (:Concept  {key, name, tenant, max_tier})

    (:Document)-[:LINKS_TO {tenant}]->(:Document)
    (:Document)-[:MENTIONS {tenant}]->(:Concept)

Two properties on that model are load-bearing for everything Phase 6 does.

**Tenant lives on every node and every relationship.** Neo4j Community
allows a single database, so tenants cannot be separated by storage and
isolation has to hold inside every individual query instead. That is the
harder version of the problem and the one worth solving: a traversal that
forgets the tenant predicate on one hop crosses the boundary silently.

**Concept keys are namespaced by tenant.** A concept node's key is
`tenant::name`, so "compensation review" in one tenant and the same phrase in
another are different nodes that can never be merged. This is the structural
answer to cross-tenant entity resolution: not a check that could be
forgotten, but an identifier that makes the mistake unrepresentable.

**Concepts carry max_tier** — the highest sensitivity of any document
mentioning them. A concept named only by restricted documents is itself a
disclosure, because its existence reveals what those documents are about.
Storing it now makes that leak measurable in Phase 6 rather than invisible.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase

from ragguard.access import TIER_RANK
from ragguard.config import settings

CONSTRAINTS = [
    ("CREATE CONSTRAINT document_uri IF NOT EXISTS "
     "FOR (d:Document) REQUIRE d.uri IS UNIQUE"),
    ("CREATE CONSTRAINT concept_key IF NOT EXISTS "
     "FOR (c:Concept) REQUIRE c.key IS UNIQUE"),
]

INDEXES = [
    "CREATE INDEX document_tenant IF NOT EXISTS FOR (d:Document) ON (d.tenant)",
    "CREATE INDEX document_tier   IF NOT EXISTS FOR (d:Document) ON (d.tier)",
    "CREATE INDEX concept_tenant  IF NOT EXISTS FOR (c:Concept)  ON (c.tenant)",
]


@contextmanager
def graph_driver() -> Iterator[Driver]:
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        yield driver
    finally:
        driver.close()


def apply_schema(driver: Driver) -> None:
    with driver.session() as session:
        for statement in CONSTRAINTS + INDEXES:
            session.run(statement)


def wipe(driver: Driver) -> None:
    """Drop all graph data.

    The graph is derived entirely from Postgres, so rebuilding is always
    correct and incremental updates would only be a source of drift between
    the two stores at this stage.
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def concept_key(tenant: str, name: str) -> str:
    """Namespaced identity — the reason cross-tenant merging cannot happen."""
    return f"{tenant}::{name}"


def higher_tier(left: str, right: str) -> str:
    return left if TIER_RANK[left] >= TIER_RANK[right] else right
