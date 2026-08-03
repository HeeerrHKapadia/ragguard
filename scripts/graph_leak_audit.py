"""Measure the leaks a document-level permission check cannot see.

Graph retrieval reports a leak rate of 0.0%. Every returned document is one
the user may read. This asks the questions that metric does not:

  How many result paths passed through a document the user cannot read?
  How many touched a concept whose every source document is forbidden?
  How many documents' existence is inferable from missing neighbours?

Run:  uv run python scripts/graph_leak_audit.py
"""

from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import load_principals
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect
from ragguard.eval.dataset import load
from ragguard.graph.audit import AUDIT_CYPHER, DEGREE_CYPHER, AuditResult
from ragguard.graph.filters import visibility_params
from ragguard.graph.retrieve import SEED_K
from ragguard.graph.store import graph_driver
from ragguard.retrieval.dense import PreFilterRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
SAMPLE = 120


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    everything = load(GOLDENS)
    cases = everything[:: max(1, len(everything) // SAMPLE)][:SAMPLE]

    try:
        with connect() as conn, graph_driver() as driver:
            driver.verify_connectivity()
            cur = conn.cursor()
            principals = load_principals(cur)
            seeder = PreFilterRetriever(conn)

            by_persona: dict[str, AuditResult] = {}

            with driver.session() as session:
                for case in cases:
                    principal = principals.get(case.persona)
                    if principal is None:
                        continue

                    seeds = seeder.retrieve(case, principal, SEED_K)
                    if not seeds:
                        continue

                    params = visibility_params(principal)
                    params["seeds"] = [
                        {"uri": d.uri, "rank": i} for i, d in enumerate(seeds, start=1)
                    ]

                    result = by_persona.setdefault(case.persona, AuditResult())

                    for record in session.run(AUDIT_CYPHER, **params):
                        result.paths += 1
                        blocked_docs = record["blocked_docs"]
                        blocked_concepts = record["blocked_concepts"]
                        if blocked_docs:
                            result.transit_paths += 1
                            result.blocked_documents.update(blocked_docs)
                        if blocked_concepts:
                            result.concept_paths += 1
                            result.blocked_concepts.update(blocked_concepts)

                    degree = session.run(DEGREE_CYPHER, **params).single()
                    if degree and degree["hidden_neighbours"]:
                        result.hidden_neighbours += degree["hidden_neighbours"]

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print(f"\n{len(cases)} sampled cases, document-level leak rate 0.0%\n")
    print(f"  {'persona':<28} {'paths':>7} {'transit':>9} {'concept':>9} {'inferable':>10}")
    print("  " + "-" * 66)

    totals = AuditResult()
    for persona in sorted(by_persona):
        r = by_persona[persona]
        totals.paths += r.paths
        totals.transit_paths += r.transit_paths
        totals.concept_paths += r.concept_paths
        totals.blocked_documents |= r.blocked_documents
        totals.blocked_concepts |= r.blocked_concepts
        totals.hidden_neighbours += r.hidden_neighbours

        print(f"  {persona:<28} {r.paths:>7} {r.transit_rate:>8.1%} "
              f"{r.concept_rate:>8.1%} {r.hidden_neighbours:>10}")

    print("\n  " + "-" * 66)
    print(f"  {'all':<28} {totals.paths:>7} {totals.transit_rate:>8.1%} "
          f"{totals.concept_rate:>8.1%} {totals.hidden_neighbours:>10}")

    print(f"\n  distinct forbidden documents on result paths : {len(totals.blocked_documents)}")
    print(f"  distinct forbidden concepts touched           : {len(totals.blocked_concepts)}")

    if totals.blocked_concepts:
        print("\n  Concepts whose every source document is forbidden to the asker.")
        print("  Each name is a summary of material the user may not read:\n")
        for name in sorted(totals.blocked_concepts)[:8]:
            print(f"    {name}")

    print(
        "\n  None of the above moves the leak rate, because no forbidden\n"
        "  document was returned. The forbidden documents were what decided\n"
        "  which permitted documents came back.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
