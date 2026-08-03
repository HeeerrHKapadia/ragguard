"""Copy the embedded chunks from Postgres into Qdrant.

Reuses the vectors already computed rather than re-embedding, so the two
engines are compared on identical inputs and any difference is the index,
not the model.

Run:  uv run python scripts/index_qdrant.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import psycopg
from qdrant_client import models

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.config import settings
from ragguard.db import connect
from ragguard.retrieval.qdrant_store import COLLECTION, client, ensure_collection

BATCH = 500


def main() -> int:
    try:
        with connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            total = cur.fetchone()[0]
            if total == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            qc = client()
            ensure_collection(qc, settings.embedding_dim)

            cur.execute(
                """SELECT c.id, c.embedding, d.source_uri, t.slug, d.section, d.sensitivity
                     FROM chunks c
                     JOIN documents d ON d.id = c.document_id
                     JOIN tenants   t ON t.id = d.tenant_id
                    WHERE c.embedding IS NOT NULL
                 ORDER BY c.id"""
            )

            started = time.time()
            pending: list[models.PointStruct] = []
            written = 0

            for chunk_id, embedding, uri, tenant, section, tier in cur:
                pending.append(models.PointStruct(
                    id=str(chunk_id),
                    # pgvector's Vector exposes to_list(), not numpy's tolist().
                    vector=embedding.to_list(),
                    payload={"uri": uri, "tenant": tenant,
                             "section": section, "tier": tier},
                ))
                if len(pending) >= BATCH:
                    qc.upsert(collection_name=COLLECTION, points=pending, wait=False)
                    written += len(pending)
                    pending = []

            if pending:
                qc.upsert(collection_name=COLLECTION, points=pending, wait=True)
                written += len(pending)

            # The last batch waits, but earlier ones did not; block until the
            # collection reports the expected count so the benchmark cannot
            # start against a half-built index.
            deadline = time.time() + 120
            while time.time() < deadline:
                count = qc.count(COLLECTION, exact=True).count
                if count >= total:
                    break
                time.sleep(1)

            elapsed = time.time() - started
            count = qc.count(COLLECTION, exact=True).count
            print(f"\n  {written} vectors in {elapsed:.1f}s ({written / elapsed:.0f}/s)")
            print(f"  collection holds {count} of {total} expected\n")

            if count != total:
                print("  PROBLEM: indexed count does not match Postgres.\n")
                return 1

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
