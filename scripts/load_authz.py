"""Load the policy into OpenFGA as relationship tuples.

Rebuilt from Postgres every run. OpenFGA is configured with in-memory
storage precisely so it cannot become a second source of truth that drifts
from the relational one.

Writes the store id to .authz_store so other scripts can find it.

Run:  uv run python scripts/load_authz.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import time

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.authz.client import Authz
from ragguard.authz.model import document_tuples, identity_tuples, tier_chain
from ragguard.config import PROJECT_ROOT
from ragguard.corpus import load_config
from ragguard.db import connect

STORE_FILE = PROJECT_ROOT / ".authz_store"


async def main() -> int:
    cfg = load_config()

    try:
        with connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT t.slug, d.source_uri, d.section, d.sensitivity
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id
                 ORDER BY t.slug, d.source_uri"""
            )
            rows = cur.fetchall()
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    if not rows:
        print("Seed the database first: uv run python scripts/seed.py")
        return 1

    by_tenant: dict[str, list[tuple[str, str, str]]] = {}
    for tenant, uri, section, tier in rows:
        by_tenant.setdefault(tenant, []).append((uri, section, tier))

    tuples = []
    for tenant_cfg in cfg["tenants"]:
        tenant = tenant_cfg["slug"]
        tuples += tier_chain(tenant)
        tuples += identity_tuples(
            tenant, tenant_cfg.get("groups", []), tenant_cfg.get("users", [])
        )
        tuples += document_tuples(tenant, by_tenant.get(tenant, []))

    started = time.time()
    async with Authz() as authz:
        store_id = await authz.create_store()
        model_id = await authz.write_model()
        written = await authz.write_tuples(tuples)

    STORE_FILE.write_text(f"{store_id}\n{model_id}\n", encoding="utf-8")

    elapsed = time.time() - started
    print(f"\n  store {store_id}")
    print(f"  model {model_id}")
    print(f"  {written} tuples in {elapsed:.1f}s ({written / elapsed:.0f}/s)")
    print(f"\n  written to {STORE_FILE.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
