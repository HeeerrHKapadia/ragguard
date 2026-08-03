"""Does routing authorization through OpenFGA cost retrieval quality?

Phase 2's pre-filter encodes the policy as a SQL predicate the application
owns. This asks the authorization service instead, then hands its answer to
the index as a filter.

If the two produce identical results, the service can be made authoritative
for free — which is the point of adopting one. Any gap is the price of that
authority, and worth knowing before paying it.

Run:  uv run python scripts/authz_retrieval_benchmark.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import load_principals
from ragguard.authz.client import Authz
from ragguard.authz.model import scoped
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect
from ragguard.eval.dataset import load
from ragguard.eval.metrics import Report
from ragguard.eval.runner import load_corpus_index, run
from ragguard.retrieval.authz_filtered import AuthzPreFilterRetriever, build_allowed_map
from ragguard.retrieval.dense import PostFilterRetriever, PreFilterRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
STORE_FILE = PROJECT_ROOT / ".authz_store"


def line(label: str, report: Report, elapsed: float, n: int) -> None:
    by_class = report.recall_by_class()
    print(f"  {label:<32} {report.recall_vs_ceiling:>10.1%} "
          f"{by_class['local']:>9.1%} {by_class['global']:>9.1%} "
          f"{report.leak_rate:>7.1%} {n / elapsed:>8.0f}")


async def main() -> int:
    if not STORE_FILE.exists():
        print("Load the tuples first: uv run python scripts/load_authz.py")
        return 1
    store_id, model_id = STORE_FILE.read_text(encoding="utf-8").split()

    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1
    cases = load(GOLDENS)

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            if cur.fetchone()[0] == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            # One ListObjects per principal, exactly as a real service would
            # do once per request.
            started = time.perf_counter()
            async with Authz(store_id=store_id, model_id=model_id) as authz:
                objects_by_user = {
                    email: await authz.list_objects(
                        scoped("user", p.tenant_slug, email), "viewer", "document"
                    )
                    for email, p in principals.items()
                }
            authz_ms = (time.perf_counter() - started) * 1000

            allowed = build_allowed_map(principals, objects_by_user, set(corpus))
            resolved = sum(len(v) for v in allowed.values())
            expected = sum(len(v) for v in objects_by_user.values())

            print(f"\n  {len(principals)} ListObjects calls in {authz_ms:.0f} ms "
                  f"({authz_ms / len(principals):.1f} ms each)")
            print(f"  {resolved} of {expected} object ids mapped back to documents")
            if resolved != expected:
                print("  WARNING: some ids did not map — sanitisation collision?")

            print(f"\n  {len(cases)} cases\n")
            print(f"  {'':<32} {'vs ceiling':>10} {'local':>9} {'global':>9} "
                  f"{'leak':>7} {'q/s':>8}")
            print("  " + "-" * 80)

            for label, retriever in (
                ("SQL predicate pre-filter", PreFilterRetriever(conn)),
                ("OpenFGA pre-filter", AuthzPreFilterRetriever(conn, allowed)),
                ("post-filter (the usual advice)", PostFilterRetriever(conn)),
            ):
                start = time.time()
                report = run(retriever, cases, principals, corpus)
                line(label, report, time.time() - start, len(cases))

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
