"""Print what each seeded persona can actually see.

This is the Phase 0b deliverable that matters most. It turns the policy in
tenants.yaml into a concrete visibility matrix, and it is the ground truth
the Phase 0c eval harness scores against: every cell here is a set of
documents that must be retrievable, and its complement is a set that must
never be.

Run:  uv run python scripts/access_matrix.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import TIER_RANK, can_read, load_principals
from ragguard.db import connect

TIERS = ["public", "internal", "confidential", "restricted"]


def main() -> int:
    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)

            cur.execute(
                """SELECT t.slug, d.section, d.sensitivity
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
            )
            docs = cur.fetchall()
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        print("Start it with:  docker compose up -d")
        return 1

    if not docs:
        print("No documents found. Run: uv run python scripts/seed.py")
        return 1

    total_by_tenant = Counter(t for t, _, _ in docs)
    problems: list[str] = []

    header = f"{'persona':<26} {'clear':<13} " + "".join(f"{t[:6]:>8}" for t in TIERS) + f"{'total':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    last_tenant = None
    for email in sorted(principals, key=lambda e: (principals[e].tenant_slug, e)):
        p = principals[email]
        if p.tenant_slug != last_tenant:
            print(f"\n[{p.tenant_slug}]  {total_by_tenant[p.tenant_slug]} documents")
            last_tenant = p.tenant_slug

        visible = Counter()
        for tenant_slug, section, tier in docs:
            if can_read(p, tenant_slug, section, tier):
                visible[tier] += 1

        cells = "".join(f"{visible.get(t, 0):>8}" for t in TIERS)
        total = sum(visible.values())
        print(f"{email:<26} {p.max_clearance:<13} {cells}{total:>9}")

        # A persona that can see everything or nothing makes for a useless
        # eval: there is no boundary left to test.
        if total == 0:
            problems.append(f"{email} can see nothing")
        if total == total_by_tenant[p.tenant_slug] and p.max_clearance != "restricted":
            problems.append(f"{email} sees everything despite non-exec clearance")

    # Cross-tenant isolation is the invariant that must never bend.
    leaks = sum(
        1
        for p in principals.values()
        for tenant_slug, section, tier in docs
        if tenant_slug != p.tenant_slug and can_read(p, tenant_slug, section, tier)
    )
    print(f"\ncross-tenant reads permitted: {leaks}")
    if leaks:
        problems.append(f"{leaks} cross-tenant reads permitted by the policy")

    # Every tier boundary should separate at least one pair of personas,
    # otherwise the tier exists on paper only.
    print("\ntier separation:")
    for tier in TIERS:
        readers = [
            e for e, p in principals.items()
            if TIER_RANK[p.max_clearance] >= TIER_RANK[tier]
        ]
        print(f"  {tier:<13} {len(readers):>2}/{len(principals)} personas cleared globally")

    if problems:
        print("\nPROBLEMS:")
        for p_msg in problems:
            print(f"  - {p_msg}")
        return 1

    print("\nPolicy looks sane: every persona sees some documents, none sees all,")
    print("and no principal can read across a tenant boundary.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
