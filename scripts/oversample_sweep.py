"""Find the conditions under which post-filtering actually collapses.

Phase 2 predicted that retrieve-then-drop would gut recall for low-privilege
users. At the default 6x oversample it did not — post-filtering came out
slightly ahead of no filtering at all. A prediction that fails is worth more
than one that is quietly dropped, so this measures the thing properly.

The mechanism only bites when the candidate pool runs out. Post-filtering
asks the database for `k * oversample` chunks with no idea who is asking,
then discards the ones the user may not see. A new hire entitled to 7 of 114
documents keeps only a sliver of that pool, so the fewer candidates fetched,
the sooner they run dry. Pre-filtering spends every slot on permitted rows
and is unaffected.

Sweeping the oversample factor turns the prediction into a curve: at what
ratio does the shortcut start costing users their results?

Run:  uv run python scripts/oversample_sweep.py
"""

from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import load_principals
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect
from ragguard.eval.dataset import load
from ragguard.eval.runner import load_corpus_index, run
from ragguard.retrieval.dense import PostFilterRetriever, PreFilterRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"
FACTORS = [1, 2, 3, 6, 12, 25, 50, 100, 200]

# The least and most privileged persona in the tenant with the sharpest
# privilege gradient: 7 visible documents against 114.
LOW = "newhire@sourcegraph.test"
HIGH = "exec@sourcegraph.test"


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    cases = [c for c in load(GOLDENS) if c.tenant == "sourcegraph"]

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            if cur.fetchone()[0] == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            print(f"\n{len(cases)} sourcegraph cases")
            print(f"low  privilege: {LOW}   (7 of 114 documents visible)")
            print(f"high privilege: {HIGH}  (all 114 visible)\n")

            print(f"  {'oversample':<12} {'post low':>10} {'post high':>11} "
                  f"{'pre low':>10} {'gap':>8}")
            print("  " + "-" * 54)

            for factor in FACTORS:
                post = run(PostFilterRetriever(conn, factor), cases, principals, corpus)
                pre = run(PreFilterRetriever(conn, factor), cases, principals, corpus)

                post_by = post.recall_by_persona()
                pre_by = pre.recall_by_persona()

                post_low = post_by.get(LOW, 0.0)
                post_high = post_by.get(HIGH, 0.0)
                pre_low = pre_by.get(LOW, 0.0)
                gap = pre_low - post_low

                flag = "  <-- collapse" if gap > 0.15 else ""
                print(f"  {factor:<12} {post_low:>9.1%} {post_high:>10.1%} "
                      f"{pre_low:>9.1%} {gap:>+7.1%}{flag}")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print(
        "\n  'gap' is what the low-privilege user loses by filtering after\n"
        "  retrieval instead of during it. The high-privilege column barely\n"
        "  moves, which is exactly why this failure survives code review:\n"
        "  whoever builds it is almost never the one it happens to.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
