"""Seed tenants, identity, and the document corpus into Postgres.

Idempotent: every write is an upsert keyed on a natural unique constraint,
so re-running updates in place instead of duplicating. That matters because
this script will be re-run constantly as the policy in tenants.yaml evolves,
and a seed script you're afraid to run twice is a seed script you stop
trusting.

Run:  uv run python scripts/seed.py
"""

from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.config import settings
from ragguard.corpus import Document, build_corpus
from ragguard.db import connect


def upsert_tenant(cur: psycopg.Cursor, slug: str, name: str) -> str:
    cur.execute(
        """INSERT INTO tenants (slug, name) VALUES (%s, %s)
           ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name
           RETURNING id""",
        (slug, name),
    )
    return cur.fetchone()[0]


def upsert_groups(cur: psycopg.Cursor, tenant_id: str, groups: list[dict]) -> dict[str, str]:
    ids: dict[str, str] = {}
    for group in groups:
        cur.execute(
            """INSERT INTO groups (tenant_id, slug, name, clearance, elevated_sections)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (tenant_id, slug) DO UPDATE
                 SET name              = EXCLUDED.name,
                     clearance         = EXCLUDED.clearance,
                     elevated_sections = EXCLUDED.elevated_sections
               RETURNING id""",
            (
                tenant_id,
                group["slug"],
                group["name"],
                group.get("clearance", "internal"),
                group.get("elevated", []),
            ),
        )
        ids[group["slug"]] = cur.fetchone()[0]
    return ids


def upsert_users(
    cur: psycopg.Cursor, tenant_id: str, users: list[dict], group_ids: dict[str, str]
) -> int:
    for user in users:
        cur.execute(
            """INSERT INTO users (tenant_id, email, display_name, title)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (tenant_id, email) DO UPDATE
                 SET display_name = EXCLUDED.display_name,
                     title        = EXCLUDED.title
               RETURNING id""",
            (tenant_id, user["email"], user["name"], user.get("title")),
        )
        user_id = cur.fetchone()[0]

        # Replace memberships rather than merging. If a group is removed from
        # the config, the membership must actually disappear — a stale grant
        # that survives a policy change is exactly the revocation bug this
        # project exists to catch.
        cur.execute("DELETE FROM user_groups WHERE user_id = %s", (user_id,))
        for slug in user.get("groups", []):
            if slug not in group_ids:
                raise KeyError(f"user {user['email']} references unknown group '{slug}'")
            cur.execute(
                "INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s)",
                (user_id, group_ids[slug]),
            )
    return len(users)


def upsert_documents(cur: psycopg.Cursor, tenant_id: str, docs: list[Document]) -> None:
    cur.executemany(
        """INSERT INTO documents
             (tenant_id, source_uri, title, section, sensitivity, content_hash)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (tenant_id, source_uri) DO UPDATE
             SET title        = EXCLUDED.title,
                 section      = EXCLUDED.section,
                 sensitivity  = EXCLUDED.sensitivity,
                 content_hash = EXCLUDED.content_hash""",
        [
            (tenant_id, d.source_uri, d.title, d.section, d.tier, d.content_hash)
            for d in docs
        ],
    )


def main() -> int:
    print(f"\nSeeding {settings.safe_dsn}\n")

    cfg, corpora = build_corpus()

    try:
        with connect() as conn:
            cur = conn.cursor()
            for tenant in cfg["tenants"]:
                slug = tenant["slug"]
                docs = corpora[slug]

                tenant_id = upsert_tenant(cur, slug, tenant["name"])
                group_ids = upsert_groups(cur, tenant_id, tenant.get("groups", []))
                n_users = upsert_users(cur, tenant_id, tenant.get("users", []), group_ids)
                upsert_documents(cur, tenant_id, docs)

                print(
                    f"  {slug:<12} {len(group_ids)} groups, {n_users} users, "
                    f"{len(docs)} documents"
                )
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"  Could not reach Postgres: {str(exc).strip()}")
        print("  Start it with:  docker compose up -d")
        return 1

    print("\nSeed complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
