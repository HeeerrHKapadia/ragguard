"""Calibrate the router, then measure what routing is worth.

Two questions, in order.

First: can the query class be predicted from the shape of the dense score
distribution, without the label? Sweeping the threshold gives the accuracy
curve and the operating point, rather than a number picked by eye.

Second: does routing actually pay? Reranking gains ten points on global
queries and costs 174x throughput, so it is only affordable if it runs on
the fraction of queries that benefit. A router that is right 70% of the time
still beats reranking everything, and the comparison has to include the
misroutes to be honest.

Run:  uv run python scripts/router_benchmark.py
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
from ragguard.retrieval.dense import PreFilterRetriever
from ragguard.retrieval.rerank import RerankingRetriever
from ragguard.retrieval.router import PROBE_K, RoutedRetriever, classify

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
THRESHOLDS = [0.030, 0.040, 0.050, 0.055, 0.060, 0.070, 0.090]

# Reranking every case would take roughly an hour on CPU. A fixed slice keeps
# every configuration on identical cases.
SAMPLE = 200


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    everything = load(GOLDENS)

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            if cur.fetchone()[0] == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            base = PreFilterRetriever(conn)

            # --- calibration: probe every case once, reuse for all thresholds
            print(f"\nProbing {len(everything)} cases for score spread...")
            probes = []
            for case in everything:
                principal = principals.get(case.persona)
                if principal is None:
                    continue
                results = base.retrieve(case, principal, PROBE_K)
                probes.append((case.query_class, results))

            print(f"\n  {'threshold':>10} {'overall':>9} {'local':>9} {'global':>9} "
                  f"{'routed to rerank':>18}")
            print("  " + "-" * 60)

            best = (None, -1.0)
            for threshold in THRESHOLDS:
                hits = {"local": [0, 0], "global": [0, 0]}
                to_global = 0
                for actual, results in probes:
                    predicted = classify(results, threshold).query_class
                    hits[actual][1] += 1
                    if predicted == actual:
                        hits[actual][0] += 1
                    if predicted == "global":
                        to_global += 1

                total_right = sum(v[0] for v in hits.values())
                total = sum(v[1] for v in hits.values())
                overall = total_right / total
                if overall > best[1]:
                    best = (threshold, overall)

                print(f"  {threshold:>10.3f} {overall:>8.1%} "
                      f"{hits['local'][0] / max(hits['local'][1], 1):>8.1%} "
                      f"{hits['global'][0] / max(hits['global'][1], 1):>8.1%} "
                      f"{to_global / total:>17.1%}")

            # Accuracy is the wrong objective on an 81%-local dataset, and
            # printing it alone would flatter a router that has learned
            # nothing. The comparison that matters is against always
            # guessing the majority class.
            majority = sum(1 for actual, _ in probes if actual == "local") / len(probes)
            print(f"\n  best threshold: {best[0]} at {best[1]:.1%} overall accuracy")
            print(f"  always guessing 'local':    {majority:.1%}")
            if best[1] <= majority:
                print("\n  The router is no better than the trivial baseline. Score\n"
                      "  spread does not separate these classes, so the accuracy\n"
                      "  figure above is majority-class bias rather than signal.")

            # --- what routing is worth ------------------------------------
            cases = everything[:: max(1, len(everything) // SAMPLE)][:SAMPLE]
            reranker = RerankingRetriever(base, conn)

            print(f"\n  Routing value, {len(cases)} sampled cases\n")
            print(f"  {'':<26} {'local':>9} {'global':>9} {'q/s':>9}")
            print("  " + "-" * 56)

            for label, retriever in (
                ("dense only", base),
                ("rerank everything", reranker),
                ("routed", RoutedRetriever(base, reranker, threshold=best[0])),
            ):
                started = time.time()
                report = run(retriever, cases, principals, corpus)
                elapsed = time.time() - started
                by_class = report.recall_by_class()
                print(f"  {label:<26} {by_class['local']:>8.1%} "
                      f"{by_class['global']:>8.1%} {len(cases) / elapsed:>8.1f}")
                if isinstance(retriever, RoutedRetriever):
                    acc = retriever.accuracy()
                    print(f"  {'':<26} router accuracy "
                          f"{acc.get('overall', 0):.1%} "
                          f"(local {acc.get('local', 0):.1%}, "
                          f"global {acc.get('global', 0):.1%})")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
