"""Attack the system deliberately. Every success becomes a permanent test.

The suite is built around one rule: an attack that has never been tried is
not a defence that works. Each check below either fails the build or is
recorded as a known-open risk with a reason.

There is no generation step in this system, so classic prompt injection has
no target — the model that would obey the instruction does not exist. That
is stated rather than quietly skipped, because "we tested for injection" and
"there is nothing here to inject into" are different claims. The attacks
that *do* have a target are the ones against retrieval and identity.

  A1  cross-tenant extraction by query crafting
  A2  tier escalation using known titles of forbidden documents
  A3  existence disclosure through result counts
  A4  stale permissions after revocation
  A5  graph transit through forbidden documents
  A6  concept nodes bridging tenants
  A7  identifier collisions in the authorization store
  A8  a single document dominating every result set

Run:  uv run python scripts/redteam.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from collections import Counter

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import can_read, load_principals
from ragguard.authz.client import Authz
from ragguard.authz.model import Tuple, safe_id, scoped
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect
from ragguard.eval.dataset import GoldenCase
from ragguard.graph.audit import AUDIT_CYPHER
from ragguard.graph.filters import visibility_params
from ragguard.graph.retrieve import GraphRetriever
from ragguard.graph.store import graph_driver
from ragguard.retrieval.dense import PreFilterRetriever

STORE_FILE = PROJECT_ROOT / ".authz_store"

PASS = "  [ok]    "
FAIL = "  [BREACH]"
OPEN = "  [open]  "


class Findings:
    def __init__(self) -> None:
        self.breaches: list[str] = []
        self.open_risks: list[str] = []

    def ok(self, name: str, detail: str = "") -> None:
        print(f"{PASS} {name}{f' — {detail}' if detail else ''}")

    def breach(self, name: str, detail: str) -> None:
        print(f"{FAIL} {name} — {detail}")
        self.breaches.append(f"{name}: {detail}")

    def open_risk(self, name: str, detail: str) -> None:
        """A real weakness that is understood and deliberately not fixed."""
        print(f"{OPEN} {name} — {detail}")
        self.open_risks.append(f"{name}: {detail}")


def probe(query: str, persona: str, tenant: str) -> GoldenCase:
    return GoldenCase(
        case_id="redteam", tenant=tenant, persona=persona,
        query=query, query_class="local", relevant_uris=(),
    )


def a1_cross_tenant(f: Findings, conn, principals) -> None:
    """Query using another tenant's most distinctive vocabulary."""
    cur = conn.cursor()
    cur.execute(
        """SELECT t.slug, d.title FROM documents d
             JOIN tenants t ON t.id = d.tenant_id
            WHERE d.sensitivity IN ('confidential', 'restricted')"""
    )
    titles_by_tenant: dict[str, list[str]] = {}
    for tenant, title in cur.fetchall():
        titles_by_tenant.setdefault(tenant, []).append(title)

    retriever = PreFilterRetriever(conn)
    violations = 0
    attempts = 0

    for email, principal in sorted(principals.items()):
        for tenant, titles in titles_by_tenant.items():
            if tenant == principal.tenant_slug:
                continue
            for title in titles[:5]:
                attempts += 1
                for doc in retriever.retrieve(probe(title, email, tenant), principal, 10):
                    if doc.tenant != principal.tenant_slug:
                        violations += 1

    if violations:
        f.breach("A1 cross-tenant extraction", f"{violations} foreign documents returned")
    else:
        f.ok("A1 cross-tenant extraction", f"{attempts} crafted queries, 0 foreign documents")


def a2_tier_escalation(f: Findings, conn, principals) -> None:
    """Search for a document by its exact title, knowing it is forbidden.

    The most realistic attack in the set: an employee hears a document
    mentioned in a meeting and searches for it by name.
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT t.slug, d.source_uri, d.section, d.sensitivity, d.title
             FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
    )
    documents = cur.fetchall()
    retriever = PreFilterRetriever(conn)

    violations = 0
    attempts = 0
    for email, principal in sorted(principals.items()):
        forbidden = [
            (uri, title) for tenant, uri, section, tier, title in documents
            if tenant == principal.tenant_slug
            and not can_read(principal, tenant, section, tier)
        ]
        for uri, title in forbidden[:12]:
            attempts += 1
            got = retriever.retrieve(probe(title, email, principal.tenant_slug), principal, 10)
            if any(doc.uri == uri for doc in got):
                violations += 1

    if violations:
        f.breach("A2 tier escalation by title", f"{violations} forbidden documents surfaced")
    else:
        f.ok("A2 tier escalation by title", f"{attempts} exact-title searches, 0 hits")


def a3_existence_disclosure(f: Findings, conn, principals) -> None:
    """Does result volume reveal how much a user is not being shown?"""
    retriever = PreFilterRetriever(conn)
    queries = ["compensation review process", "security incident response",
               "board meeting minutes", "hiring plan headcount"]

    counts: dict[str, list[int]] = {}
    for email, principal in sorted(principals.items()):
        got = [
            len(retriever.retrieve(probe(q, email, principal.tenant_slug), principal, 10))
            for q in queries
        ]
        counts[email] = got

    # Compare the least and most privileged persona in one tenant. A
    # consistent shortfall is a signal about the size of what is hidden.
    low = counts.get("newhire@sourcegraph.test", [])
    high = counts.get("exec@sourcegraph.test", [])
    if low and high and sum(low) < sum(high):
        f.open_risk(
            "A3 existence disclosure",
            f"low-privilege returns {sum(low)} results where exec returns {sum(high)}; "
            "count alone signals hidden volume",
        )
    else:
        f.ok("A3 existence disclosure", "result counts do not separate privilege levels")


async def a4_stale_permissions(f: Findings, principals) -> None:
    """After revoking membership, does access stop immediately?"""
    if not STORE_FILE.exists():
        f.open_risk("A4 stale permissions", "authorization store not loaded, skipped")
        return

    store_id, model_id = STORE_FILE.read_text(encoding="utf-8").split()
    victim = "eng@gitlab.test"
    principal = principals.get(victim)
    if principal is None:
        f.open_risk("A4 stale permissions", "test persona missing")
        return

    user_obj = scoped("user", principal.tenant_slug, victim)
    membership = Tuple(
        user=user_obj, relation="member",
        object=scoped("group", principal.tenant_slug, "engineering"),
    )

    async with Authz(store_id=store_id, model_id=model_id) as authz:
        before = len(await authz.list_objects(user_obj, "viewer", "document"))
        await authz.delete_tuples([membership])
        after = len(await authz.list_objects(user_obj, "viewer", "document"))
        await authz.write_tuples([membership])
        restored = len(await authz.list_objects(user_obj, "viewer", "document"))

    if after >= before:
        f.breach("A4 stale permissions", f"access unchanged after revocation ({after})")
    elif restored != before:
        f.breach("A4 stale permissions", f"restore incomplete: {restored} vs {before}")
    else:
        f.ok("A4 stale permissions", f"{before} -> {after} on revoke, restored cleanly")


def a5_graph_transit(f: Findings, conn, principals, driver) -> None:
    """Does guarded traversal still route through forbidden documents?"""
    retriever = PreFilterRetriever(conn)
    queries = ["compensation bands", "incident escalation", "board strategy"]

    unguarded_violations = 0
    guarded_violations = 0

    with driver.session() as session:
        for email, principal in sorted(principals.items()):
            for query in queries:
                seeds = retriever.retrieve(
                    probe(query, email, principal.tenant_slug), principal, 10
                )
                if not seeds:
                    continue
                params = visibility_params(principal)
                params["seeds"] = [
                    {"uri": d.uri, "rank": i} for i, d in enumerate(seeds, start=1)
                ]
                for record in session.run(AUDIT_CYPHER, **params):
                    if record["blocked_docs"]:
                        unguarded_violations += 1

    # The guarded retriever must produce no path with a forbidden interior;
    # its Cypher enforces that, so any result here is a regression.
    for email, principal in sorted(principals.items()):
        graph = GraphRetriever(retriever, driver, guarded=True)
        for query in queries:
            for doc in graph.retrieve(
                probe(query, email, principal.tenant_slug), principal, 10
            ):
                if not can_read(principal, doc.tenant, doc.section, doc.tier):
                    guarded_violations += 1

    if guarded_violations:
        f.breach("A5 graph transit", f"{guarded_violations} forbidden documents returned")
    else:
        f.ok("A5 graph transit",
             f"guard holds ({unguarded_violations} paths would leak unguarded)")


def a6_concept_bridging(f: Findings, driver) -> None:
    """Can a concept node connect two tenants?"""
    with driver.session() as session:
        shared = session.run(
            """MATCH (c:Concept)<-[:MENTIONS]-(d:Document)
               WITH c, collect(DISTINCT d.tenant) AS tenants
               WHERE size(tenants) > 1
               RETURN count(c) AS n"""
        ).single()["n"]
        crossing = session.run(
            "MATCH (a)-[r]->(b) WHERE a.tenant <> b.tenant RETURN count(r) AS n"
        ).single()["n"]

    if shared or crossing:
        f.breach("A6 concept tenant bridging",
                 f"{shared} shared concepts, {crossing} crossing relationships")
    else:
        f.ok("A6 concept tenant bridging", "no concept or edge spans tenants")


def a7_identifier_collision(f: Findings, conn) -> None:
    """Do two document URIs sanitise to the same authorization object?

    Introduced in Phase 7: OpenFGA ids cannot contain a colon, so URIs are
    rewritten. Rewriting is lossy, and two documents mapping to one object
    would share permissions — the higher-privileged one silently granting
    access to the other.
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT t.slug, d.source_uri FROM documents d
             JOIN tenants t ON t.id = d.tenant_id"""
    )
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []

    for tenant, uri in cur.fetchall():
        key = scoped("document", tenant, uri)
        if key in seen and seen[key] != uri:
            collisions.append((seen[key], uri))
        seen[key] = uri

    if collisions:
        detail = f"{len(collisions)} pairs, e.g. {collisions[0][0]} / {collisions[0][1]}"
        f.breach("A7 identifier collision", detail)
    else:
        f.ok("A7 identifier collision", f"{len(seen)} ids, all distinct")

    # Same question for the sanitiser in isolation, on adversarial input
    # chosen so that naive character substitution would collapse them.
    hostile = [
        "a:b.md", "a_b.md", "a|b.md", "a b.md", "a*b.md",
        "gitlab://x.md", "gitlab___x.md",
    ]
    mapped = Counter(safe_id(x) for x in hostile)
    dupes = [k for k, n in mapped.items() if n > 1]
    if dupes:
        f.breach(
            "A7 sanitiser injectivity",
            f"distinct inputs share an id ({dupes[0]})",
        )
    else:
        f.ok("A7 sanitiser injectivity",
             f"{len(hostile)} adversarial inputs, all distinct")


def a8_document_dominance(f: Findings, conn, principals) -> None:
    """Does one document appear in nearly every result set?

    A document embedded to sit near every query would dominate retrieval
    without leaking anything — a denial-of-quality attack rather than a
    disclosure. Also catches an accidental centroid-like document.
    """
    retriever = PreFilterRetriever(conn)
    queries = ["onboarding checklist", "expense policy", "deployment process",
               "performance review", "customer escalation", "security training"]

    appearances = Counter()
    total = 0
    for email, principal in sorted(principals.items()):
        for query in queries:
            got = retriever.retrieve(probe(query, email, principal.tenant_slug), principal, 10)
            if got:
                total += 1
                for doc in got:
                    appearances[doc.uri] += 1

    if not total:
        f.open_risk("A8 document dominance", "no results to analyse")
        return

    worst_uri, worst_count = appearances.most_common(1)[0]
    share = worst_count / total
    if share > 0.5:
        f.breach("A8 document dominance",
                 f"{worst_uri} appears in {share:.0%} of result sets")
    else:
        f.ok("A8 document dominance", f"top document appears in {share:.0%} of result sets")


async def main() -> int:
    findings = Findings()

    try:
        with connect() as conn, graph_driver() as driver:
            driver.verify_connectivity()
            cur = conn.cursor()
            principals = load_principals(cur)

            if not principals:
                print("Seed the database first: uv run python scripts/seed.py")
                return 1

            print(f"\nRed team, {len(principals)} personas\n")
            a1_cross_tenant(findings, conn, principals)
            a2_tier_escalation(findings, conn, principals)
            a3_existence_disclosure(findings, conn, principals)
            await a4_stale_permissions(findings, principals)
            a5_graph_transit(findings, conn, principals, driver)
            a6_concept_bridging(findings, driver)
            a7_identifier_collision(findings, conn)
            a8_document_dominance(findings, conn, principals)

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    if findings.open_risks:
        print(f"{len(findings.open_risks)} known-open risk(s), documented and accepted:")
        for risk in findings.open_risks:
            print(f"    {risk}")
        print()

    if findings.breaches:
        print(f"{len(findings.breaches)} BREACH(ES):")
        for breach in findings.breaches:
            print(f"    {breach}")
        print()
        return 1

    print("No breaches. Every attack above is now a permanent regression test.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
