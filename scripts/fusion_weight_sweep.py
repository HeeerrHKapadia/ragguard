"""Does the lexical channel earn its place in the fusion?

Adding hybrid search made things worse: 82.6% of ceiling against the
dense-only pre-filter's 86.8%, with local recall dropping five points. The
received wisdom is that hybrid retrieval beats either channel alone, so
either the wisdom does not apply here or the implementation is wrong.

Sweeping the fusion weights answers it directly. At weight (1, 0) only the
dense ranking contributes and the result should match dense-only retrieval;
at (0, 1) only lexical does. If the best score sits at one end, that channel
is carrying the other and the fusion is costing accuracy and latency for
nothing.

Worth stating the likely cause up front so the measurement is honest about
what it is testing: the golden queries are derived from document titles, and
the terms are OR-joined for the lexical query. A five-term OR matches any
chunk containing any one of those words, so the lexical ranking may be
mostly noise that equal-weight fusion then lets outvote a good dense result.

Run:  uv run python scripts/fusion_weight_sweep.py
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
from ragguard.retrieval.hybrid import HybridRetriever

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"

# (dense, lexical)
WEIGHTS = [
    (1.0, 0.0),
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 1.0),
    (0.5, 1.0),
    (0.0, 1.0),
]


def main() -> int:
    if not GOLDENS.exists():
        print("Build the goldens first: uv run python scripts/build_goldens.py")
        return 1

    cases = load(GOLDENS)

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)

            cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            if cur.fetchone()[0] == 0:
                print("Index the corpus first: uv run python scripts/index_corpus.py")
                return 1

            print(f"\n{len(cases)} cases\n")
            print(f"  {'dense':>6} {'lexical':>8} {'vs ceiling':>12} "
                  f"{'local':>9} {'global':>9} {'leak':>7}")
            print("  " + "-" * 55)

            best = {"overall": (None, -1.0), "local": (None, -1.0), "global": (None, -1.0)}

            for dense_w, lex_w in WEIGHTS:
                report = run(
                    HybridRetriever(conn, weights=(dense_w, lex_w)),
                    cases, principals, corpus,
                )
                by_class = report.recall_by_class()
                scores = {
                    "overall": report.recall_vs_ceiling,
                    "local": by_class["local"],
                    "global": by_class["global"],
                }
                for key, value in scores.items():
                    if value > best[key][1]:
                        best[key] = ((dense_w, lex_w), value)

                print(f"  {dense_w:>6.2f} {lex_w:>8.2f} {scores['overall']:>11.1%} "
                      f"{by_class['local']:>8.1%} {by_class['global']:>8.1%} "
                      f"{report.leak_rate:>6.1%}")

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    for key in ("overall", "local", "global"):
        (dw, lw), value = best[key]
        print(f"  best {key:<8} dense={dw}, lexical={lw}   {value:.1%}")

    local_w = best["local"][0]
    global_w = best["global"][0]

    if local_w != global_w:
        print(
            "\n  The optimum depends on the query class, so there is no single\n"
            "  right answer to freeze. Lexical matching earns its place on\n"
            "  cross-document questions, where a shared term is often the only\n"
            "  thing linking two documents, and costs accuracy on point lookups,\n"
            "  where the dense ranking already had it right and the lexical\n"
            "  noise only outvotes it.\n"
            "\n  That is an argument for routing by query class rather than\n"
            "  picking one configuration and averaging the loss — which is the\n"
            "  same conclusion Phase 5 reaches about the knowledge graph, from\n"
            "  a completely different direction.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
