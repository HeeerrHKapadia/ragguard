"""pgvector against Qdrant, under permission filters of varying selectivity.

Phase 2 measured pre-filtering halving pgvector's throughput: 492 queries
per second unfiltered, 243 with a selective WHERE. The explanation offered
was structural — the HNSW index covers every row, so a filter is applied
around the search rather than inside it, and the engine walks further to
find enough surviving candidates. Qdrant indexes payload fields alongside
the vectors and evaluates conditions during traversal.

That is a vendor claim until it is measured, so this measures it.

Three things, in order:

**Parity.** Qdrant's filter is the fifth expression of the same access
policy. It must admit exactly what the oracle admits.

**Recall under filtering.** Ground truth is exact search — sequential scan,
no index, correct by definition — restricted to the same permitted set. An
approximate index that quietly loses results when filters get selective is
the failure mode worth catching, and it is invisible without a reference.

**Latency across selectivity.** The personas provide a natural gradient:
an executive sees every document in their tenant, a new hire sees 7 of 114.

Run:  uv run python scripts/qdrant_benchmark.py
"""

from __future__ import annotations

import pathlib
import statistics
import sys
import time

import psycopg
from pgvector import Vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import can_read, load_principals
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect
from ragguard.embedding import embed_query
from ragguard.eval.dataset import load
from ragguard.retrieval.filters import visibility_sql
from ragguard.retrieval.qdrant_store import COLLECTION, client, search, visibility_filter

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
QUERIES = 40
TOP_K = 10
POOL = TOP_K * 6

SELECT_COLS = """SELECT d.source_uri, c.embedding <=> %(vec)s AS distance"""
FROM_JOIN = """FROM chunks c
               JOIN documents d ON d.id = c.document_id
               JOIN tenants   t ON t.id = d.tenant_id"""


def collapse(rows, k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for uri, _score in rows:
        if uri not in seen:
            seen.add(uri)
            out.append(uri)
            if len(out) >= k:
                break
    return out


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    cases = load(GOLDENS)
    seen_q: set[tuple[str, str]] = set()
    queries: list[tuple[str, str]] = []
    for case in cases:
        key = (case.tenant, case.query)
        if key not in seen_q:
            seen_q.add(key)
            queries.append(key)
    queries = queries[:QUERIES]

    qc = client()
    if not qc.collection_exists(COLLECTION):
        print("Index Qdrant first: uv run python scripts/index_qdrant.py")
        return 1

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)

            cur.execute(
                """SELECT t.slug, d.source_uri, d.section, d.sensitivity
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
            )
            documents = cur.fetchall()

            # --- parity ---------------------------------------------------
            print("\nPolicy parity\n")
            mismatches = 0
            for email in sorted(principals):
                principal = principals[email]
                oracle = {
                    uri for tenant, uri, section, tier in documents
                    if can_read(principal, tenant, section, tier)
                }

                from_qdrant: set[str] = set()
                offset = None
                while True:
                    points, offset = qc.scroll(
                        collection_name=COLLECTION,
                        scroll_filter=visibility_filter(principal),
                        limit=2000, offset=offset, with_payload=True,
                        with_vectors=False,
                    )
                    from_qdrant.update(p.payload["uri"] for p in points)
                    if offset is None:
                        break

                extra = from_qdrant - oracle
                missing = oracle - from_qdrant
                mismatches += len(extra) + len(missing)
                status = "ok" if not (extra or missing) else "MISMATCH"
                print(f"  {email:<28} {len(oracle):>4} allowed   {status}")
                for uri in sorted(extra)[:2]:
                    print(f"      qdrant admits, oracle forbids : {uri}")
                for uri in sorted(missing)[:2]:
                    print(f"      qdrant blocks, oracle allows  : {uri}")

            if mismatches:
                print(f"\n  POLICY MISMATCH: {mismatches} disagreements.\n")
                return 1
            print("\n  Qdrant filter is identical to the reference implementation.")

            # --- recall and latency ---------------------------------------
            print("\n\nFiltered search, ground truth is exact sequential scan\n")
            print(f"  {'persona':<28} {'visible':>8} {'pg recall':>10} {'pg ms':>8} "
                  f"{'qd recall':>10} {'qd ms':>8}")
            print("  " + "-" * 78)

            embeddings = {q: Vector(embed_query(q)) for _t, q in queries}

            for email in sorted(principals):
                principal = principals[email]
                where, params = visibility_sql(principal)

                visible = sum(
                    1 for tenant, _u, section, tier in documents
                    if can_read(principal, tenant, section, tier)
                )

                pg_hits, qd_hits = [], []
                pg_times, qd_times = [], []

                for tenant, query in queries:
                    if tenant != principal.tenant_slug:
                        continue
                    vec = embeddings[query]

                    # Ground truth: sequential scan, no index involved.
                    cur.execute("SET LOCAL enable_indexscan = off")
                    cur.execute("SET LOCAL enable_bitmapscan = off")
                    cur.execute(
                        f"{SELECT_COLS} {FROM_JOIN} WHERE {where} "
                        f"ORDER BY distance LIMIT %(limit)s",
                        {**params, "vec": vec, "limit": POOL},
                    )
                    truth = set(collapse(cur.fetchall(), TOP_K))
                    cur.execute("SET LOCAL enable_indexscan = on")
                    cur.execute("SET LOCAL enable_bitmapscan = on")

                    if not truth:
                        continue

                    started = time.perf_counter()
                    cur.execute(
                        f"{SELECT_COLS} {FROM_JOIN} WHERE {where} "
                        f"ORDER BY distance LIMIT %(limit)s",
                        {**params, "vec": vec, "limit": POOL},
                    )
                    pg = collapse(cur.fetchall(), TOP_K)
                    pg_times.append((time.perf_counter() - started) * 1000)
                    pg_hits.append(len(set(pg) & truth) / len(truth))

                    started = time.perf_counter()
                    rows = search(qc, vec.to_list(), principal, POOL)
                    qd = collapse([(r[0], r[4]) for r in rows], TOP_K)
                    qd_times.append((time.perf_counter() - started) * 1000)
                    qd_hits.append(len(set(qd) & truth) / len(truth))

                if not pg_hits:
                    continue

                print(f"  {email:<28} {visible:>8} "
                      f"{statistics.mean(pg_hits):>9.1%} "
                      f"{statistics.median(pg_times):>7.1f} "
                      f"{statistics.mean(qd_hits):>9.1%} "
                      f"{statistics.median(qd_times):>7.1f}")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print(
        "\n  Recall is measured against exact search over the same permitted\n"
        "  set, so a number below 100% means the approximate index missed\n"
        "  documents it was allowed to return.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
