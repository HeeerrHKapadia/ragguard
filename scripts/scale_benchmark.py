"""Find the corpus size at which Qdrant overtakes pgvector on filtered search.

At 4,996 vectors pgvector won by 4-25x, and the query plans explained why:
under a permission filter Postgres abandons the HNSW index and exact-scans
the permitted rows. At that size an exact scan is trivially cheap, so the
comparison measured transport rather than index architecture.

Exact scan is linear in the permitted set. Qdrant's filterable HNSW is not.
Somewhere above 4,996 the lines cross, and finding that point is the only
way to turn "pgvector is faster here" into advice that generalises.

**Scaling method.** Vectors are produced by perturbing real embeddings and
renormalising, not by sampling random ones. HNSW's performance depends
heavily on how clustered the vectors are — uniformly random vectors are a
pathological case that would flatter exact scan and make the crossover
appear later than it is. Perturbation preserves the clustering of real text
embeddings while multiplying the volume.

Nothing here measures retrieval quality. Synthetic vectors have no meaning,
so recall against them would be meaningless. This measures latency only.

Run:  uv run python scripts/scale_benchmark.py
"""

from __future__ import annotations

import math
import os
import pathlib
import random
import statistics
import sys
import time

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.config import settings
from ragguard.db import connect
from ragguard.retrieval.qdrant_store import client

TABLE = "scale_chunks"
COLLECTION = "ragguard_scale"

DEFAULT_SCALES = [5_000, 25_000, 100_000, 250_000, 500_000]

# Overridable so a run can be cut short while iterating, without editing the
# file: SCALES=5000,25000 uv run python scripts/scale_benchmark.py
SCALES = [
    int(x) for x in os.getenv("SCALES", ",".join(map(str, DEFAULT_SCALES))).split(",")
]

# Two filter selectivities drawn from the real personas: an executive who
# sees everything in a tenant, and a new hire who sees a sliver.
SELECTIVITIES = {"broad (47%)": 0.47, "narrow (3.4%)": 0.034}

QUERIES = 20
TOP_K = 10
NOISE = 0.05
SEED = 20260803


def perturb(vector: list[float], rng: random.Random, noise: float = NOISE) -> list[float]:
    """A nearby vector: add gaussian noise, renormalise.

    Keeps the point inside the same cluster as its source, so the graph
    Qdrant builds resembles one built over a genuinely larger corpus rather
    than over uniform noise.
    """
    out = [v + rng.gauss(0.0, noise) for v in vector]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def build_pg(conn, seeds: list[list[float]], total: int, rng: random.Random) -> float:
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cur.execute(
        f"""CREATE TABLE {TABLE} (
              id       bigserial PRIMARY KEY,
              bucket   int NOT NULL,
              embedding vector({settings.embedding_dim})
            )"""
    )
    conn.commit()

    started = time.time()
    with cur.copy(f"COPY {TABLE} (bucket, embedding) FROM STDIN") as copy:
        for i in range(total):
            # `bucket` stands in for the permission predicate: an integer
            # 0-999 so any selectivity can be expressed as a range.
            copy.write_row((rng.randrange(1000), Vector(perturb(seeds[i % len(seeds)], rng))))
    conn.commit()

    cur.execute(f"CREATE INDEX ON {TABLE} (bucket)")
    cur.execute(
        f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = 16, ef_construction = 64)"
    )
    cur.execute(f"ANALYZE {TABLE}")
    conn.commit()
    return time.time() - started


def build_qdrant(qc: QdrantClient, seeds: list[list[float]], total: int,
                 rng: random.Random) -> float:
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    qc.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim, distance=models.Distance.COSINE
        ),
    )
    qc.create_payload_index(
        collection_name=COLLECTION, field_name="bucket",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    started = time.time()
    batch: list[models.PointStruct] = []
    for i in range(total):
        batch.append(models.PointStruct(
            id=i,
            vector=perturb(seeds[i % len(seeds)], rng),
            payload={"bucket": rng.randrange(1000)},
        ))
        if len(batch) >= 2000:
            qc.upsert(collection_name=COLLECTION, points=batch, wait=False)
            batch = []
    if batch:
        qc.upsert(collection_name=COLLECTION, points=batch, wait=True)

    deadline = time.time() + 600
    while time.time() < deadline:
        if qc.count(COLLECTION, exact=True).count >= total:
            break
        time.sleep(2)
    return time.time() - started


def time_pg(conn, probes: list[list[float]], ceiling: int) -> float:
    cur = conn.cursor()
    times = []
    for vec in probes:
        started = time.perf_counter()
        cur.execute(
            f"SELECT id FROM {TABLE} WHERE bucket < %s "
            f"ORDER BY embedding <=> %s LIMIT %s",
            (ceiling, Vector(vec), TOP_K),
        )
        cur.fetchall()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def time_qdrant(qc: QdrantClient, probes: list[list[float]], ceiling: int) -> float:
    times = []
    flt = models.Filter(must=[
        models.FieldCondition(key="bucket", range=models.Range(lt=ceiling))
    ])
    for vec in probes:
        started = time.perf_counter()
        qc.query_points(
            collection_name=COLLECTION, query=vec,
            query_filter=flt, limit=TOP_K, with_payload=False,
        )
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def main() -> int:
    rng = random.Random(SEED)

    try:
        with connect() as conn:
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 2000"
            )
            seeds = [row[0].to_list() for row in cur.fetchall()]
            if not seeds:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            probes = [perturb(rng.choice(seeds), rng, noise=0.15) for _ in range(QUERIES)]
            qc = client()

            print(f"\nScaling from {len(seeds)} real embeddings by perturbation")
            print(f"{QUERIES} probe queries, top-{TOP_K}, median latency\n")

            header = f"  {'vectors':>9} {'build pg':>9} {'build qd':>9}"
            for label in SELECTIVITIES:
                header += f" {label + ' pg':>17} {label + ' qd':>17}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            crossovers: dict[str, int] = {}

            for total in SCALES:
                pg_build = build_pg(conn, seeds, total, random.Random(SEED))
                qd_build = build_qdrant(qc, seeds, total, random.Random(SEED))

                row = f"  {total:>9,} {pg_build:>8.1f}s {qd_build:>8.1f}s"
                for label, fraction in SELECTIVITIES.items():
                    ceiling = max(1, int(1000 * fraction))
                    pg_ms = time_pg(conn, probes, ceiling)
                    qd_ms = time_qdrant(qc, probes, ceiling)
                    row += f" {pg_ms:>16.1f} {qd_ms:>16.1f}"
                    if qd_ms < pg_ms and label not in crossovers:
                        crossovers[label] = total
                print(row)

            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
            conn.commit()
            if qc.collection_exists(COLLECTION):
                qc.delete_collection(COLLECTION)

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    if crossovers:
        for label, size in crossovers.items():
            print(f"  Qdrant overtakes pgvector at ~{size:,} vectors ({label} filter)")
    else:
        print(f"  No crossover up to {SCALES[-1]:,} vectors — pgvector still ahead.")
    print(
        "\n  Latency only. The vectors are synthetic, so recall against them\n"
        "  would measure nothing.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
