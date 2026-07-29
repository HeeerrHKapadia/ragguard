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

from ragguard.corpus import build_corpus

TIERS = ["public", "internal", "confidential", "restricted"]


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

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nEvery tenant has documents in every tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
