"""What does cross-encoder reranking actually buy?

Reranking is the most consistently recommended upgrade in production RAG
guidance, usually quoted at five to fifteen points of improvement. Those
figures come from benchmarks whose queries were written by people; these
queries are derived from document titles, which is a different distribution
and may well behave differently.

Measured against the first stage it wraps, so the delta is attributable to
reranking rather than to anything underneath it.

Run:  uv run python scripts/rerank_benchmark.py
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
from ragguard.retrieval.dense import PreFilterRetriever
from ragguard.retrieval.rerank import RerankingRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"

# Reranking runs a full forward pass per candidate document, so the whole
# 884-case set would take a long while on CPU. A fixed slice keeps the
# comparison honest — both configurations see identical cases — while
# staying quick enough to iterate on.
SAMPLE = 200


def row(label: str, report: Report, elapsed: float, n: int) -> None:
    by_class = report.recall_by_class()
    print(
        f"  {label:<24} {report.recall_vs_ceiling:>9.1%} "
        f"{by_class['local']:>9.1%} {by_class['global']:>9.1%} "
        f"{report.leak_rate:>7.1%} {n / elapsed:>9.1f}"
    )


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    everything = load(GOLDENS)
    cases = everything[:: max(1, len(everything) // SAMPLE)][:SAMPLE]

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            if cur.fetchone()[0] == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            print(f"\n{len(cases)} cases sampled from {len(everything)}\n")
            print(f"  {'':<24} {'vs ceiling':>9} {'local':>9} {'global':>9} "
                  f"{'leak':>7} {'q/s':>9}")
            print("  " + "-" * 72)

            base = PreFilterRetriever(conn)
            started = time.time()
            base_report = run(base, cases, principals, corpus)
            base_t = time.time() - started
            row("dense pre-filter", base_report, base_t, len(cases))

            reranked = RerankingRetriever(base, conn)
            started = time.time()
            rr_report = run(reranked, cases, principals, corpus)
            rr_t = time.time() - started
            row("+ cross-encoder", rr_report, rr_t, len(cases))

            delta = rr_report.recall_vs_ceiling - base_report.recall_vs_ceiling
            by_base = base_report.recall_by_class()
            by_rr = rr_report.recall_by_class()

            print(f"\n  overall  {delta:+.1%}")
            print(f"  local    {by_rr['local'] - by_base['local']:+.1%}")
            print(f"  global   {by_rr['global'] - by_base['global']:+.1%}")
            print(f"  cost     {base_t / rr_t:.2f}x throughput "
                  f"({len(cases) / base_t:.0f} -> {len(cases) / rr_t:.0f} q/s)")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
