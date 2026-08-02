"""Measure what approximate nearest-neighbour search actually costs.

Phase 0a deliberately shipped without an HNSW index so that exact search —
perfect recall by definition — could serve as the reference. This script
adds the index and re-runs the identical evaluation, so the price of
approximation is a measured number rather than an assumption.

The trade is real: exact search compares the query against every stored
vector, which is correct but linear in corpus size. HNSW walks a navigable
graph and reaches a good answer without visiting everything, at the cost of
occasionally missing a true nearest neighbour.

Run:  uv run python scripts/hnsw_benchmark.py
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
from ragguard.retrieval.dense import DenseRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
INDEX_NAME = "chunks_embedding_hnsw"


def measure(conn, cases, principals, corpus) -> tuple[Report, float]:
    started = time.time()
    report = run(DenseRetriever(conn), cases, principals, corpus)
    return report, time.time() - started


def line(label: str, report: Report, elapsed: float, n: int) -> None:
    by_class = report.recall_by_class()
    print(
        f"  {label:<12} {report.recall_vs_ceiling:>7.1%} "
        f"{by_class['local']:>9.1%} {by_class['global']:>9.1%} "
        f"{elapsed:>8.1f}s {n / elapsed:>9.0f}"
    )


def main() -> int:
    if not GOLDENS.exists():
        print("Build the golden dataset first: uv run python scripts/build_goldens.py")
        return 1

    cases = load(GOLDENS)

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            n_chunks = cur.fetchone()[0]
            if n_chunks == 0:
                print("No embedded chunks. Run: uv run python scripts/index_corpus.py")
                return 1

            # --- exact ---------------------------------------------------
            cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
            conn.commit()

            print(f"\n{n_chunks} chunks, {len(cases)} cases\n")
            print(f"  {'':<12} {'vs ceil':>7} {'local':>9} {'global':>9} {'time':>9} {'q/s':>9}")
            exact, exact_t = measure(conn, cases, principals, corpus)
            line("exact", exact, exact_t, len(cases))

            # --- approximate ---------------------------------------------
            # m and ef_construction are HNSW's build-time quality knobs:
            # more connections per node and a wider search during
            # construction give a better graph and a slower build.
            build_started = time.time()
            cur.execute(
                f"""CREATE INDEX {INDEX_NAME} ON chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)"""
            )
            conn.commit()
            build_t = time.time() - build_started

            approx, approx_t = measure(conn, cases, principals, corpus)
            line("hnsw", approx, approx_t, len(cases))

            print(f"\n  index build: {build_t:.1f}s")

            recall_delta = approx.recall_vs_ceiling - exact.recall_vs_ceiling
            speed = exact_t / approx_t if approx_t else 0
            print(f"  recall change: {recall_delta:+.2%}")
            print(f"  speed change:  {speed:.2f}x")

            if abs(recall_delta) < 0.005:
                print(
                    "\n  Approximation costs essentially nothing at this corpus size.\n"
                    "  4996 vectors is small enough that HNSW reaches the true nearest\n"
                    "  neighbours anyway. The trade-off this index exists to make only\n"
                    "  becomes visible at a scale this project does not reach — worth\n"
                    "  knowing, and worth not pretending otherwise."
                )

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
