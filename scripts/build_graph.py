"""Build the knowledge graph in Neo4j from the seeded corpus.

Rebuilds from scratch every run. The graph is derived from Postgres, so a
full rebuild is always correct, whereas incremental updates would create a
second source of truth that drifts.

Run:  uv run python scripts/build_graph.py
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import TIER_RANK
from ragguard.corpus import build_corpus
from ragguard.graph.extract import extract, resolve_link
from ragguard.graph.store import apply_schema, concept_key, graph_driver, wipe

# A heading naming only one document connects nothing. A heading on hundreds
# of documents connects everything to everything, which is the same as
# connecting nothing while costing far more to traverse.
MIN_CONCEPT_DOCS = 2
MAX_CONCEPT_DOCS = 40


def main() -> int:
    _cfg, corpora = build_corpus()
    started = time.time()

    with graph_driver() as driver:
        driver.verify_connectivity()
        apply_schema(driver)
        wipe(driver)

        totals = {"documents": 0, "concepts": 0, "links": 0, "mentions": 0}

        for tenant in sorted(corpora):
            docs = corpora[tenant]
            known = {d.source_uri for d in docs}
            by_uri = {d.source_uri: d for d in docs}

            extractions = {d.source_uri: extract(d.source_uri, d.text) for d in docs}

            # --- concepts, filtered by how many documents name them ------
            concept_docs: dict[str, set[str]] = defaultdict(set)
            for uri, found in extractions.items():
                for name in found.concepts:
                    concept_docs[name].add(uri)

            keep = {
                name: uris for name, uris in concept_docs.items()
                if MIN_CONCEPT_DOCS <= len(uris) <= MAX_CONCEPT_DOCS
            }

            # A concept inherits the highest sensitivity among the documents
            # that name it. Phase 6 turns this into a measurable leak; here
            # it just has to be recorded truthfully.
            concept_tier = {
                name: max((by_uri[u].tier for u in uris), key=lambda t: TIER_RANK[t])
                for name, uris in keep.items()
            }

            # --- links, resolved to documents actually in the corpus -----
            link_edges: set[tuple[str, str]] = set()
            for uri, found in extractions.items():
                for target in found.links:
                    hit = resolve_link(target, uri, known, tenant)
                    if hit and hit != uri:
                        link_edges.add((uri, hit))

            with driver.session() as session:
                session.run(
                    """UNWIND $rows AS row
                       CREATE (d:Document {
                         uri: row.uri, tenant: row.tenant, tier: row.tier,
                         section: row.section, title: row.title
                       })""",
                    rows=[
                        {"uri": d.source_uri, "tenant": tenant, "tier": d.tier,
                         "section": d.section, "title": d.title}
                        for d in docs
                    ],
                )

                session.run(
                    """UNWIND $rows AS row
                       CREATE (c:Concept {
                         key: row.key, name: row.name,
                         tenant: row.tenant, max_tier: row.max_tier
                       })""",
                    rows=[
                        {"key": concept_key(tenant, name), "name": name,
                         "tenant": tenant, "max_tier": concept_tier[name]}
                        for name in keep
                    ],
                )

                session.run(
                    """UNWIND $rows AS row
                       MATCH (a:Document {uri: row.src})
                       MATCH (b:Document {uri: row.dst})
                       CREATE (a)-[:LINKS_TO {tenant: row.tenant}]->(b)""",
                    rows=[
                        {"src": src, "dst": dst, "tenant": tenant}
                        for src, dst in sorted(link_edges)
                    ],
                )

                session.run(
                    """UNWIND $rows AS row
                       MATCH (d:Document {uri: row.uri})
                       MATCH (c:Concept  {key: row.key})
                       CREATE (d)-[:MENTIONS {tenant: row.tenant}]->(c)""",
                    rows=[
                        {"uri": uri, "key": concept_key(tenant, name), "tenant": tenant}
                        for name, uris in keep.items() for uri in sorted(uris)
                    ],
                )

            mentions = sum(len(uris) for uris in keep.values())
            print(f"  {tenant:<14} {len(docs):>4} docs  {len(keep):>4} concepts  "
                  f"{len(link_edges):>4} links  {mentions:>5} mentions")

            totals["documents"] += len(docs)
            totals["concepts"] += len(keep)
            totals["links"] += len(link_edges)
            totals["mentions"] += mentions

        elapsed = time.time() - started
        print(f"\n  {totals['documents']} documents, {totals['concepts']} concepts, "
              f"{totals['links'] + totals['mentions']} relationships in {elapsed:.1f}s")
        print("  extraction cost: $0.00 (no LLM calls)")

        # --- the invariant that must never bend ----------------------------
        with driver.session() as session:
            crossing = session.run(
                """MATCH (a)-[r]->(b)
                   WHERE a.tenant <> b.tenant OR r.tenant <> a.tenant
                   RETURN count(r) AS n"""
            ).single()["n"]
            orphan_concepts = session.run(
                "MATCH (c:Concept) WHERE NOT (c)<-[:MENTIONS]-() RETURN count(c) AS n"
            ).single()["n"]

        print(f"\n  cross-tenant relationships: {crossing}")
        print(f"  orphaned concepts:          {orphan_concepts}")

        if crossing:
            print("\n  FAILED: a relationship crosses a tenant boundary.")
            return 1

        # --- the surface Phase 6 has to defend -----------------------------
        with driver.session() as session:
            bridging = session.run(
                """MATCH (d:Document)-[:MENTIONS]->(c:Concept)
                   WITH c, collect(DISTINCT d.tier) AS tiers
                   WHERE size(tiers) > 1
                   RETURN count(c) AS n"""
            ).single()["n"]
            sensitive = session.run(
                """MATCH (c:Concept)
                   WHERE c.max_tier IN ['confidential', 'restricted']
                   RETURN count(c) AS n"""
            ).single()["n"]
            pairs = session.run(
                """MATCH (a:Document)-[:MENTIONS]->(:Concept)<-[:MENTIONS]-(b:Document)
                   WHERE a.uri < b.uri
                   RETURN count(DISTINCT [a.uri, b.uri]) AS n"""
            ).single()["n"]

        total_concepts = totals["concepts"]
        print("\n  Leak surface for Phase 6")
        print(f"    concepts naming confidential or restricted material  {sensitive:>4}")
        print(f"    concepts spanning more than one tier                 {bridging:>4}"
              f"  ({bridging / total_concepts:.0%})")
        print(f"    document pairs reachable through a shared concept    {pairs:>4}")
        print(
            "\n    A concept spanning tiers is a path from material a user may\n"
            "    read to material they may not. Its name alone can disclose that\n"
            "    the restricted document exists and what it concerns, which no\n"
            "    document-level permission check would ever catch."
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
