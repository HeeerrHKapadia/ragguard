"""Generate the golden dataset from the corpus and seeded personas.

Writes eval/goldens.jsonl, which IS committed to git. The dataset is the
contract every later phase is measured against, so it has to be reviewable
in a diff: if a change to chunking or policy shifts the questions, that
should show up in a pull request rather than silently rescoring history.

Run:  uv run python scripts/build_goldens.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import load_principals
from ragguard.config import PROJECT_ROOT
from ragguard.corpus import build_corpus
from ragguard.db import connect
from ragguard.eval.dataset import build_cases, save

OUT = PROJECT_ROOT / "eval" / "goldens.jsonl"


def main() -> int:
    _cfg, corpora = build_corpus()

    try:
        with connect() as conn:
            principals = load_principals(conn.cursor())
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        print("Start it with:  docker compose up -d")
        return 1

    if not principals:
        print("No personas found. Run: uv run python scripts/seed.py")
        return 1

    by_tenant: dict[str, list[str]] = {}
    for email, principal in principals.items():
        by_tenant.setdefault(principal.tenant_slug, []).append(email)
    for emails in by_tenant.values():
        emails.sort()

    cases = build_cases(corpora, by_tenant)
    save(cases, OUT)

    classes = Counter(c.query_class for c in cases)
    distinct = len({(c.tenant, c.query_class, c.query) for c in cases})

    print(f"\nWrote {len(cases)} cases to {OUT.relative_to(PROJECT_ROOT)}")
    print(f"  distinct queries : {distinct}")
    print(f"  local cases      : {classes['local']}")
    print(f"  global cases     : {classes['global']}")

    print("\nBy tenant:")
    for tenant in sorted({c.tenant for c in cases}):
        local = sum(1 for c in cases if c.tenant == tenant and c.query_class == "local")
        glob = sum(1 for c in cases if c.tenant == tenant and c.query_class == "global")
        print(f"  {tenant:<14} {local + glob:>5}  ({local} local, {glob} global)")

    if classes["global"] == 0:
        print("\nPROBLEM: no global cases — the GraphRAG comparison needs them.")
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
