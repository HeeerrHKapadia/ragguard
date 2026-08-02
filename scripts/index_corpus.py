"""Chunk every document, embed the chunks, and store them.

Run after seed.py, which creates the document rows this fills in against.

Run:  uv run python scripts/index_corpus.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import psycopg
from pgvector import Vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.chunking import chunk_text, estimate_tokens
from ragguard.config import settings
from ragguard.corpus import build_corpus
from ragguard.db import connect
from ragguard.embedding import embed_documents

BATCH = 256


def main() -> int:
    print(f"\nIndexing into {settings.safe_dsn}")
    print(f"Model: {settings.embedding_model} ({settings.embedding_dim} dims)\n")

    _cfg, corpora = build_corpus()

    try:
        with connect() as conn:
            cur = conn.cursor()

            cur.execute(
                """SELECT d.source_uri, d.id, d.tenant_id
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
            )
            doc_ids = {uri: (doc_id, tenant_id) for uri, doc_id, tenant_id in cur.fetchall()}

            if not doc_ids:
                print("No documents found. Run: uv run python scripts/seed.py")
                return 1

            # Chunks are derived data, fully rebuilt from documents each run.
            # Upserting them individually would leave orphans behind whenever
            # the chunk size changes and a document yields fewer pieces than
            # it did last time.
            cur.execute("DELETE FROM chunks")

            total_chunks = 0
            started = time.time()

            for tenant_slug in sorted(corpora):
                pending: list[tuple] = []
                tenant_chunks = 0

                for doc in corpora[tenant_slug]:
                    entry = doc_ids.get(doc.source_uri)
                    if entry is None:
                        continue
                    doc_id, tenant_id = entry

                    for chunk in chunk_text(doc.text):
                        pending.append((doc_id, tenant_id, chunk.ordinal, chunk.text))

                    if len(pending) >= BATCH:
                        tenant_chunks += flush(cur, pending)
                        pending = []

                tenant_chunks += flush(cur, pending)
                total_chunks += tenant_chunks
                print(f"  {tenant_slug:<14} {tenant_chunks:>5} chunks")

            conn.commit()

            elapsed = time.time() - started
            print(f"\n{total_chunks} chunks in {elapsed:.1f}s ({total_chunks / elapsed:.0f}/s)")

            cur.execute("SELECT count(*), count(embedding) FROM chunks")
            rows, embedded = cur.fetchone()
            print(f"stored: {rows} chunks, {embedded} with embeddings")
            if rows != embedded:
                print("PROBLEM: some chunks have no embedding")
                return 1

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        print("Start it with:  docker compose up -d")
        return 1

    print()
    return 0


def flush(cur: psycopg.Cursor, pending: list[tuple]) -> int:
    """Embed a batch and write it.

    Embedding in batches rather than one row at a time matters: the model
    amortises setup across the batch, and the difference is roughly an order
    of magnitude on a corpus this size.
    """
    if not pending:
        return 0

    vectors = embed_documents([text for _, _, _, text in pending])

    cur.executemany(
        """INSERT INTO chunks
             (document_id, tenant_id, ordinal, text, token_count, embedding)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        [
            (doc_id, tenant_id, ordinal, text, estimate_tokens(text), Vector(vec))
            for (doc_id, tenant_id, ordinal, text), vec in zip(pending, vectors)
        ],
    )
    return len(pending)


if __name__ == "__main__":
    raise SystemExit(main())
