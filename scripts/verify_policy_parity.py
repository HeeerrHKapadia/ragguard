"""Prove the SQL policy agrees with the reference implementation.

Phase 0b kept access.py deliberately slow and obvious so that a fast path
could be diffed against something known-correct. This is that diff.

Every persona is checked against every document — 12 x 639 comparisons — and
the SQL predicate must return exactly the set the oracle permits. Not
approximately, not for a sample: exactly, for all of them.

Two disagreements are possible and they are not equally bad. SQL admitting
something the oracle forbids is a leak. SQL rejecting something the oracle
allows is over-blocking. Both are reported, because a policy that silently
hides material a user is entitled to is a broken policy that no leak metric
will ever catch.

Run:  uv run python scripts/verify_policy_parity.py
"""

from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import can_read, load_principals
from ragguard.db import connect
from ragguard.retrieval.filters import visibility_sql


def main() -> int:
    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)

            cur.execute(
                """SELECT d.source_uri, t.slug, d.section, d.sensitivity
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
            )
            documents = cur.fetchall()

            if not documents or not principals:
                print("Seed the database first: uv run python scripts/seed.py")
                return 1

            print(f"\n{len(principals)} personas x {len(documents)} documents "
                  f"= {len(principals) * len(documents)} comparisons\n")

            total_leaks = 0
            total_blocks = 0

            for email in sorted(principals):
                principal = principals[email]

                oracle = {
                    uri for uri, tenant, section, tier in documents
                    if can_read(principal, tenant, section, tier)
                }

                where, params = visibility_sql(principal)
                cur.execute(
                    f"""SELECT d.source_uri
                          FROM documents d JOIN tenants t ON t.id = d.tenant_id
                         WHERE {where}""",
                    params,
                )
                from_sql = {row[0] for row in cur.fetchall()}

                leaks = from_sql - oracle       # SQL too permissive
                blocks = oracle - from_sql      # SQL too strict

                total_leaks += len(leaks)
                total_blocks += len(blocks)

                status = "ok" if not (leaks or blocks) else "MISMATCH"
                print(f"  {email:<28} {len(oracle):>4} visible   {status}")

                for uri in sorted(leaks)[:3]:
                    print(f"      SQL admits, oracle forbids : {uri}")
                for uri in sorted(blocks)[:3]:
                    print(f"      SQL blocks, oracle allows  : {uri}")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    if total_leaks or total_blocks:
        print(f"POLICY MISMATCH: {total_leaks} over-permissive, {total_blocks} over-strict")
        print("The oracle is correct by definition. Fix the SQL.")
        return 1

    print("SQL policy is identical to the reference implementation.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
