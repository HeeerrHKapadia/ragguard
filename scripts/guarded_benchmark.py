"""Does guarding the traversal close the leak, and what does it cost?

The audit found 5.8% of result paths transiting documents the user may not
read — 64.3% for the least privileged persona — while the document-level
leak rate stayed at 0.0%.

Guarding checks every node on a path rather than only its destination. That
must drive transit violations to zero. It will also remove paths, because a
route through forbidden material is exactly the kind a sparse graph relies
on, so the recall cost has to be measured rather than hoped about.

The number to watch is not the average. It is whether the cost falls hardest
on the personas who were leaking most — which would mean the fix protects
low-privilege users by giving them worse results, trading one inequity for
another.

Run:  uv run python scripts/guarded_benchmark.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import load_principals
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect
from ragguard.eval.dataset import load
from ragguard.eval.runner import load_corpus_index, run
from ragguard.graph.audit import AUDIT_CYPHER
from ragguard.graph.filters import visibility_params
from ragguard.graph.retrieve import GUARDED_EXPAND_CYPHER, SEED_K, GraphRetriever
from ragguard.graph.store import graph_driver
from ragguard.retrieval.dense import PreFilterRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
SAMPLE = 120

# Confirms guarding worked, by re-running the audit against only the paths
# the guarded traversal would actually keep.
GUARDED_AUDIT = GUARDED_EXPAND_CYPHER.replace(
    """RETURN n.uri              AS uri,
       count(DISTINCT s.uri) AS seed_hits,
       min(seed.rank)     AS best_rank
ORDER BY seed_hits DESC, best_rank ASC
LIMIT $limit""",
    "RETURN count(*) AS kept",
)


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    everything = load(GOLDENS)
    sampled = everything[:: max(1, len(everything) // SAMPLE)][:SAMPLE]

    try:
        with connect() as conn, graph_driver() as driver:
            driver.verify_connectivity()
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)
            seeder = PreFilterRetriever(conn)

            # --- did guarding close the leak? -----------------------------
            print(f"\nAuditing {len(sampled)} cases\n")
            print(f"  {'persona':<28} {'paths':>8} {'transit':>9} {'kept':>8} {'lost':>8}")
            print("  " + "-" * 64)

            stats: dict[str, dict[str, int]] = {}
            with driver.session() as session:
                for case in sampled:
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
                    params["limit"] = 10_000

                    row = stats.setdefault(
                        case.persona, {"paths": 0, "transit": 0, "kept": 0}
                    )
                    for record in session.run(AUDIT_CYPHER, **params):
                        row["paths"] += 1
                        if record["blocked_docs"]:
                            row["transit"] += 1

                    kept = session.run(GUARDED_AUDIT, **params).single()
                    row["kept"] += kept["kept"] if kept else 0

            total = {"paths": 0, "transit": 0, "kept": 0}
            for persona in sorted(stats):
                row = stats[persona]
                for key in total:
                    total[key] += row[key]
                rate = row["transit"] / row["paths"] if row["paths"] else 0
                lost = 1 - (row["kept"] / row["paths"]) if row["paths"] else 0
                print(f"  {persona:<28} {row['paths']:>8} {rate:>8.1%} "
                      f"{row['kept']:>8} {lost:>7.1%}")

            rate = total["transit"] / total["paths"] if total["paths"] else 0
            lost = 1 - (total["kept"] / total["paths"]) if total["paths"] else 0
            print("  " + "-" * 64)
            print(f"  {'all':<28} {total['paths']:>8} {rate:>8.1%} "
                  f"{total['kept']:>8} {lost:>7.1%}")

            # The guard must remove exactly the transit-violating paths and
            # nothing else. More removed would mean it is over-blocking;
            # fewer would mean a violating path survived. Either is a bug,
            # and this is the regression gate that catches it.
            expected_kept = total["paths"] - total["transit"]
            if total["kept"] != expected_kept:
                print(f"\n  GUARD FAILED: kept {total['kept']}, expected {expected_kept}")
                return 1
            print("\n  Guard removes exactly the violating paths: transit is now 0.")

            # --- what did it cost in recall? ------------------------------
            print(f"\n\nRetrieval quality, {len(everything)} cases\n")
            print(f"  {'':<24} {'vs ceiling':>11} {'local':>9} {'global':>9} "
                  f"{'leak':>7} {'q/s':>8}")
            print("  " + "-" * 72)

            for label, guarded in (("graph, unguarded", False), ("graph, guarded", True)):
                retriever = GraphRetriever(seeder, driver, guarded=guarded)
                started = time.time()
                report = run(retriever, everything, principals, corpus)
                elapsed = time.time() - started
                by_class = report.recall_by_class()
                print(f"  {label:<24} {report.recall_vs_ceiling:>10.1%} "
                      f"{by_class['local']:>8.1%} {by_class['global']:>8.1%} "
                      f"{report.leak_rate:>6.1%} {len(everything) / elapsed:>8.0f}")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
