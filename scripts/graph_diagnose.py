"""Why did graph expansion barely help?

The benchmark says +0.6% on global queries, which is indistinguishable from
nothing. That result is only useful if the reason is known, because "graphs
do not help here" and "this graph does not encode what these queries need"
lead to completely different next steps.

Global queries in this dataset are section-shaped: the relevant set for a
global case is the documents of one handbook section. So the question is
whether graph edges connect documents within a section, or across sections.

An edge that crosses sections is not wrong — a link from an engineering page
to a finance page is a real relationship — but it is actively unhelpful for
a query whose correct answer is one section, because expansion spends its
budget leaving the region it should be filling in.

Run:  uv run python scripts/graph_diagnose.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.config import PROJECT_ROOT
from ragguard.eval.dataset import load
from ragguard.graph.store import graph_driver

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"


def main() -> int:
    cases = load(GOLDENS)
    globals_by_query = {}
    for case in cases:
        if case.query_class == "global":
            globals_by_query[(case.tenant, case.query)] = set(case.relevant_uris)

    with graph_driver() as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            print("\n=== do edges stay inside a section? ===\n")

            for rel in ("LINKS_TO",):
                record = session.run(
                    f"""MATCH (a:Document)-[:{rel}]->(b:Document)
                        RETURN sum(CASE WHEN a.section = b.section THEN 1 ELSE 0 END) AS same,
                               count(*) AS total"""
                ).single()
                same, total = record["same"], record["total"]
                print(f"  {rel:<12} {same}/{total} same-section  "
                      f"({same / total:.0%})" if total else f"  {rel}: none")

            record = session.run(
                """MATCH (a:Document)-[:MENTIONS]->(:Concept)<-[:MENTIONS]-(b:Document)
                   WHERE a.uri < b.uri
                   RETURN sum(CASE WHEN a.section = b.section THEN 1 ELSE 0 END) AS same,
                          count(*) AS total"""
            ).single()
            same, total = record["same"], record["total"]
            print(f"  {'shared concept':<12} {same}/{total} same-section  "
                  f"({same / total:.0%})")

            print("\n=== can traversal even reach the answer? ===\n")
            print(f"  {'tenant':<14} {'queries':>8} {'reachable':>11} {'of relevant':>13}")
            print("  " + "-" * 50)

            by_tenant: dict[str, list[float]] = {}
            for (tenant, _query), relevant in globals_by_query.items():
                uris = sorted(relevant)
                # Take one document as a seed and see how much of the rest of
                # the section is within two hops. Generous: a real query does
                # not get handed a correct seed.
                record = session.run(
                    """MATCH (s:Document {uri: $seed})
                       OPTIONAL MATCH (s)-[:LINKS_TO|MENTIONS*1..2]-(n:Document)
                       WHERE n.uri IN $targets
                       RETURN count(DISTINCT n.uri) AS reached""",
                    seed=uris[0], targets=uris[1:],
                ).single()
                reached = record["reached"]
                denom = len(uris) - 1
                if denom > 0:
                    by_tenant.setdefault(tenant, []).append(reached / denom)

            for tenant in sorted(by_tenant):
                values = by_tenant[tenant]
                mean = sum(values) / len(values)
                print(f"  {tenant:<14} {len(values):>8} {mean:>10.0%} {'':>13}")

            overall = [v for values in by_tenant.values() for v in values]
            if overall:
                print(f"\n  overall: {sum(overall) / len(overall):.0%} of a section's other "
                      f"documents are within two hops of one of its members")

    print(
        "\n  If that number is low, the graph does not encode section\n"
        "  membership, and no amount of weight tuning will make expansion\n"
        "  find section-shaped answers. The edges are real; they just point\n"
        "  somewhere else.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
