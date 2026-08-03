"""Does the graph beat the control group, and on which queries?

Phase 3 froze the numbers this has to clear:

    local   94.6%  dense pre-filter
    global  32.9%  dense pre-filter
    global  42.5%  dense pre-filter + cross-encoder rerank

The last one is the honest bar. Reranking already recovered most of the
global gap using nothing exotic, and a graph that only beats the naive
baseline has beaten a straw man.

Reported per query class, never as an aggregate. An average over a corpus
that is 81% local queries would let a large global gain vanish into a small
local loss, or the reverse — and the whole point is finding out which
queries the graph is for.

Run:  uv run python scripts/graph_benchmark.py
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
from ragguard.eval.metrics import Report
from ragguard.eval.runner import load_corpus_index, run
from ragguard.graph.retrieve import GraphRetriever
from ragguard.graph.store import graph_driver
from ragguard.retrieval.dense import PreFilterRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"

CONTROL = {"local": 0.946, "global": 0.329}
CONTROL_RERANKED_GLOBAL = 0.425


def row(label: str, report: Report, elapsed: float, n: int) -> dict:
    by_class = report.recall_by_class()
    print(
        f"  {label:<28} {report.recall_vs_ceiling:>9.1%} "
        f"{by_class['local']:>9.1%} {by_class['global']:>9.1%} "
        f"{report.leak_rate:>7.1%} {n / elapsed:>8.0f}"
    )
    return by_class


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    cases = load(GOLDENS)

    try:
        with connect() as conn, graph_driver() as driver:
            driver.verify_connectivity()
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            if cur.fetchone()[0] == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            with driver.session() as session:
                nodes = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            if nodes == 0:
                print("Build the graph first: uv run python scripts/build_graph.py")
                return 1

            print(f"\n{len(cases)} cases, {nodes} graph nodes\n")
            print(f"  {'':<28} {'vs ceiling':>9} {'local':>9} {'global':>9} "
                  f"{'leak':>7} {'q/s':>8}")
            print("  " + "-" * 76)

            base = PreFilterRetriever(conn)
            started = time.time()
            base_report = run(base, cases, principals, corpus)
            base_t = time.time() - started
            base_class = row("dense pre-filter (control)", base_report, base_t, len(cases))

            results = {}
            for weights, label in (
                ((1.0, 1.0), "graph, equal weight"),
                ((2.0, 1.0), "graph, seed-weighted"),
                ((1.0, 2.0), "graph, expansion-weighted"),
            ):
                retriever = GraphRetriever(base, driver, weights=weights)
                started = time.time()
                report = run(retriever, cases, principals, corpus)
                elapsed = time.time() - started
                results[label] = row(label, report, elapsed, len(cases))

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print(f"\n  Control to beat:  local {CONTROL['local']:.1%}, "
          f"global {CONTROL['global']:.1%}")
    print(f"  With reranking:   global {CONTROL_RERANKED_GLOBAL:.1%}")

    print("\n  Change against the dense control, per class:\n")
    for label, by_class in results.items():
        d_local = by_class["local"] - base_class["local"]
        d_global = by_class["global"] - base_class["global"]
        beats = "yes" if by_class["global"] > CONTROL_RERANKED_GLOBAL else "no"
        print(f"    {label:<28} local {d_local:+6.1%}   global {d_global:+6.1%}"
              f"   beats reranking: {beats}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
