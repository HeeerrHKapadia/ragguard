"""Phase 0a deliverable: prove the database actually works.

Checks, in order:
  1. We can connect at all.
  2. The pgvector extension is installed.
  3. Every expected table exists.
  4. We can write the full identity + corpus chain.
  5. Vector similarity search returns the CORRECT ORDER (not just "some rows").
  6. Full-text search works — the lexical half of Phase 3's hybrid retrieval.
  7. A tenant filter genuinely isolates tenants.

Everything runs inside one transaction that gets rolled back, so the script
is repeatable and leaves no residue.

Run:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import psycopg
from pgvector import Vector

# Make `src/` importable when running this file directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.config import settings
from ragguard.db import connect

EXPECTED_TABLES = {
    "tenants", "users", "groups", "user_groups", "documents", "chunks",
}

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def unit_vector(dim: int, components: dict[int, float]) -> Vector:
    """Build a normalized vector with only `components` non-zero.

    Deterministic vectors let us assert on the ORDER of similarity results.
    Random vectors would only prove the query ran, not that it ranked right.

    Returns pgvector's Vector type, NOT a plain list. psycopg adapts a list
    as a Postgres array (double precision[]), which the <=> operator rejects.
    Inserts happen to work via an implicit assignment cast, which makes this
    an easy bug to ship: writes succeed, reads blow up.
    """
    vec = [0.0] * dim
    for index, value in components.items():
        vec[index] = value
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        raise ValueError("zero vector cannot be normalized")
    return Vector([v / norm for v in vec])


class Checker:
    """Collects pass/fail results so one failure doesn't abort the rest."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, ok: bool, label: str, detail: str = "") -> bool:
        suffix = f" — {detail}" if detail else ""
        # flush=True so progress appears live even when output is piped.
        print(f"{PASS if ok else FAIL} {label}{suffix}", flush=True)
        if not ok:
            self.failures.append(label)
        return ok


def run_checks(conn: psycopg.Connection, check: Checker) -> None:
    dim = settings.embedding_dim
    cur = conn.cursor()

    # --- 1. server version ------------------------------------------------
    cur.execute("SHOW server_version")
    check(True, "connected", f"Postgres {cur.fetchone()[0]}")

    # --- 2. extensions ----------------------------------------------------
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    check(row is not None, "pgvector installed", f"v{row[0]}" if row else "MISSING")

    # --- 3. schema --------------------------------------------------------
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    found = {r[0] for r in cur.fetchall()}
    missing = EXPECTED_TABLES - found
    if not check(not missing, "schema present",
                 f"{len(EXPECTED_TABLES)} tables" if not missing
                 else f"missing {sorted(missing)}"):
        print("\n  The init scripts only run on an EMPTY data volume.")
        print("  If you edited db/init/ after first start, reset with:")
        print("      docker compose down -v")
        print("      docker compose up -d")
        return

    # --- 4. write the full chain ------------------------------------------
    cur.execute(
        "INSERT INTO tenants (slug, name) VALUES (%s, %s) RETURNING id",
        ("acme-smoke", "Acme (smoke test)"),
    )
    tenant_a = cur.fetchone()[0]

    cur.execute(
        "INSERT INTO tenants (slug, name) VALUES (%s, %s) RETURNING id",
        ("globex-smoke", "Globex (smoke test)"),
    )
    tenant_b = cur.fetchone()[0]

    cur.execute(
        """INSERT INTO users (tenant_id, email, display_name, title)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (tenant_a, "ana@acme.test", "Ana Ruiz", "Staff Engineer"),
    )
    user_id = cur.fetchone()[0]

    cur.execute(
        "INSERT INTO groups (tenant_id, slug, name) VALUES (%s, %s, %s) RETURNING id",
        (tenant_a, "engineering", "Engineering"),
    )
    group_id = cur.fetchone()[0]

    cur.execute(
        "INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s)",
        (user_id, group_id),
    )
    check(True, "identity chain written", "tenant -> user -> group")

    cur.execute(
        """INSERT INTO documents (tenant_id, source_uri, title, sensitivity)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (tenant_a, "smoke://handbook/oncall", "On-call Handbook", "internal"),
    )
    doc_a = cur.fetchone()[0]

    cur.execute(
        """INSERT INTO documents (tenant_id, source_uri, title, sensitivity)
           VALUES (%s, %s, %s, %s) RETURNING id""",
        (tenant_b, "smoke://secret/comp", "Compensation Bands", "confidential"),
    )
    doc_b = cur.fetchone()[0]

    # Deterministic geometry:
    #   near     -> identical to the query   => cosine distance 0.0
    #   middling -> 45 degrees off           => 0.2929
    #   far      -> orthogonal               => 1.0
    near = unit_vector(dim, {0: 1.0})
    middling = unit_vector(dim, {0: 1.0, 1: 1.0})
    far = unit_vector(dim, {1: 1.0})

    rows = [
        (near, "Escalate a Sev-1 incident to the on-call engineer immediately."),
        (middling, "Rotate the on-call schedule every Monday morning."),
        (far, "The cafeteria serves lunch between noon and two."),
    ]
    for ordinal, (vec, text) in enumerate(rows):
        cur.execute(
            """INSERT INTO chunks (document_id, tenant_id, ordinal, text, embedding)
               VALUES (%s, %s, %s, %s, %s)""",
            (doc_a, tenant_a, ordinal, text, vec),
        )

    # Same embedding as tenant A's best match, but confidential and in the
    # OTHER tenant. This is the whole project in miniature.
    cur.execute(
        """INSERT INTO chunks (document_id, tenant_id, ordinal, text, embedding)
           VALUES (%s, %s, %s, %s, %s)""",
        (doc_b, tenant_b, 0, "Staff engineer band is 180k to 240k base.", near),
    )
    check(True, "corpus written", "2 tenants, 2 docs, 4 chunks")

    # --- 5. vector similarity ORDER ---------------------------------------
    # `<=>` is pgvector's cosine distance operator: 0 = identical.
    cur.execute(
        """SELECT ordinal, embedding <=> %s AS distance
           FROM chunks
           WHERE tenant_id = %s
           ORDER BY distance
           LIMIT 3""",
        (near, tenant_a),
    )
    results = cur.fetchall()
    order = [r[0] for r in results]
    distances = [round(float(r[1]), 4) for r in results]
    check(order == [0, 1, 2], "similarity ranks correctly",
          f"ordinals {order}, distances {distances}")

    # Assert the actual geometry, not merely the ordering.
    check(abs(distances[0] - 0.0) < 1e-6, "identical vector -> distance 0", str(distances[0]))
    check(abs(distances[1] - 0.2929) < 1e-3, "45-degree vector -> distance 0.293",
          str(distances[1]))
    check(abs(distances[2] - 1.0) < 1e-6, "orthogonal vector -> distance 1", str(distances[2]))

    # --- 6. full-text search ----------------------------------------------
    cur.execute(
        """SELECT ordinal FROM chunks
           WHERE tenant_id = %s AND tsv @@ plainto_tsquery('english', %s)""",
        (tenant_a, "escalate incident"),
    )
    ft_hits = [r[0] for r in cur.fetchall()]
    check(ft_hits == [0], "full-text search works", f"matched ordinals {ft_hits}")

    # --- 7. tenant isolation ------------------------------------------------
    # Unfiltered, the confidential Globex row is a PERFECT match for Ana's
    # query. Relevance alone will happily leak it.
    cur.execute("SELECT count(*) FROM chunks WHERE embedding <=> %s < 0.001", (near,))
    unfiltered = cur.fetchone()[0]

    cur.execute(
        "SELECT count(*) FROM chunks WHERE tenant_id = %s AND embedding <=> %s < 0.001",
        (tenant_a, near),
    )
    filtered = cur.fetchone()[0]

    check(unfiltered == 2 and filtered == 1, "tenant filter isolates",
          f"{unfiltered} perfect matches unfiltered, {filtered} within tenant")


def main() -> int:
    check = Checker()
    print(f"\nConnecting to {settings.safe_dsn}\n", flush=True)

    try:
        # The `with` must be INSIDE the try: @contextmanager is lazy, so
        # calling connect() alone runs none of its body and can never raise.
        with connect() as conn:
            try:
                run_checks(conn, check)
            finally:
                # Roll back whether checks passed or blew up, so the script
                # is always repeatable.
                conn.rollback()
                print("\n  (rolled back — database left clean)")
    except psycopg.OperationalError as exc:
        print(f"{FAIL} connect")
        print(f"\n  Could not reach Postgres: {str(exc).strip()}")
        print("  Is the container running?   docker compose ps")
        print("  Start it with:              docker compose up -d")
        return 1

    print()
    if check.failures:
        print(f"FAILED: {len(check.failures)} check(s) — {', '.join(check.failures)}\n")
        return 1

    print("All checks passed. Phase 0a infrastructure is working.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
