"""Dry-run the corpus build and print what would be ingested.

Runs no database work. The point is to see the tier distribution before
committing to it: if a tier is empty, the leak tests that depend on it
would silently pass for the wrong reason.

Run:  uv run python scripts/corpus_report.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import BASE_CLEARANCE, Grant, Principal, can_read
from ragguard.corpus import build_corpus, load_config

TIERS = ["public", "internal", "confidential", "restricted"]


def principals_from_config(cfg: dict) -> dict[str, list[Principal]]:
    """Rebuild the personas from config, without needing the database.

    Lets a dry run answer the question that actually matters — does each
    persona see a different corpus — before anything is seeded or embedded.
    """
    out: dict[str, list[Principal]] = {}
    for tenant in cfg["tenants"]:
        groups = {g["slug"]: g for g in tenant.get("groups", [])}
        people = []
        for user in tenant.get("users", []):
            grants = tuple(
                Grant(
                    clearance=groups[slug].get("clearance", "internal"),
                    elevated_sections=tuple(groups[slug].get("elevated", [])),
                )
                for slug in user.get("groups", [])
                if slug in groups
            ) or (Grant(clearance=BASE_CLEARANCE),)
            people.append(
                Principal(email=user["email"], tenant_slug=tenant["slug"], grants=grants)
            )
        out[tenant["slug"]] = people
    return out


def main() -> int:
    _cfg, corpora = build_corpus()

    grand = Counter()
    problems: list[str] = []

    for slug, docs in corpora.items():
        tiers = Counter(d.tier for d in docs)
        grand.update(tiers)
        chars = sum(len(d.text) for d in docs)

        print(f"\n{slug}  —  {len(docs)} docs, {chars // 1000}k chars")
        print(f"  sections: {len({d.section for d in docs})}")
        for tier in TIERS:
            count = tiers.get(tier, 0)
            flag = "   <-- EMPTY" if count == 0 else ""
            print(f"  {tier:<13} {count:>4}{flag}")
            if count == 0:
                problems.append(f"{slug} has no {tier} documents")

    print(f"\n{'=' * 46}")
    total = sum(grand.values())
    print(f"total: {total} docs across {len(corpora)} tenants")
    for tier in TIERS:
        share = grand.get(tier, 0) / total * 100 if total else 0
        print(f"  {tier:<13} {grand.get(tier, 0):>4}  ({share:.0f}%)")

    # Tier counts alone are not enough. A tenant can have every tier
    # non-empty and still hand two adjacent personas an identical corpus,
    # which is the demo failing silently. This checks the thing that
    # actually matters.
    print(f"\n{'=' * 46}")
    print("Documents visible per persona\n")

    cfg = load_config()
    for tenant, people in principals_from_config(cfg).items():
        docs = corpora.get(tenant, [])
        counts = []
        for person in people:
            visible = sum(
                1 for d in docs if can_read(person, tenant, d.section, d.tier)
            )
            counts.append((person.email, visible))

        rendered = "  ".join(f"{e.split('@')[0]}={n}" for e, n in counts)
        print(f"  {tenant:<14} {rendered}")

        # Adjacent personas returning the same number is the signal that a
        # privilege level has nothing of its own left.
        seen: dict[int, str] = {}
        for email, n in counts:
            if n in seen:
                problems.append(
                    f"{tenant}: {seen[n]} and {email.split('@')[0]} both see {n} "
                    "documents — the persona contrast has collapsed"
                )
            seen[n] = email.split("@")[0]

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nEvery tenant has all four tiers, and every persona sees a different corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
