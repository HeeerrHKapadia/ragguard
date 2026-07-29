"""Score retrievers against the golden dataset.

With no real retriever yet, this runs the three reference implementations to
calibrate the harness. Their correct scores are known in advance, so this is
really a test of the scorer: if `leaky` does not report a catastrophic leak
rate, or `oracle` does not report a clean sweep, the measuring instrument is
broken and nothing it says later can be trusted.

Run:  uv run python scripts/evaluate.py
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
from ragguard.eval.metrics import Report
from ragguard.eval.retriever import LeakyRetriever, NullRetriever, OracleRetriever
from ragguard.eval.runner import load_corpus_index, run

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"

# What each reference retriever MUST score. These are not aspirations; they
# follow from the definitions. A mismatch means the harness is wrong.
#
# Measured against `recall_vs_ceiling`, not raw recall: raw recall cannot
# reach 1.0 when a global query has more relevant documents than k, and
# arithmetic applies to oracles too.
#
# Only the oracle is required to hit the ceiling exactly. Leaky lands just
# below it, and the reason is worth stating: when a case has more relevant
# documents than k, leaky spends some of its slots on forbidden ones while
# the oracle spends all of them on permitted ones. Same k, better used.
TOL = 1e-6
EXPECTED = {
    "null":   {"leak": ("==", 0.0), "ceiling": ("==", 0.0)},
    "leaky":  {"leak": (">=", 0.5), "ceiling": (">=", 0.95)},
    "oracle": {"leak": ("==", 0.0), "ceiling": ("==", 1.0)},
}


def summarize(name: str, report: Report) -> None:
    by_class = report.recall_by_class()
    print(f"\n{name}")
    print(f"  leak rate         {report.leak_rate:>8.1%}   ({report.leaked_doc_count} docs leaked)")
    print(f"  allowed recall    {report.allowed_recall:>8.1%}   (ceiling {report.ceiling_recall:.1%})")
    print(f"  recall / ceiling  {report.recall_vs_ceiling:>8.1%}")
    print(f"  over-block rate   {report.over_block_rate:>8.1%}")
    print(f"  recall  local     {by_class['local']:>8.1%}")
    print(f"  recall  global    {by_class['global']:>8.1%}")

    parity = report.recall_parity()
    if parity:
        cells = "  ".join(f"{t}={v:.2f}" for t, v in parity.items())
        print(f"  recall parity     {cells}")


def assert_metric(name: str, label: str, actual: float, want: tuple[str, float]) -> str | None:
    op, target = want
    ok = abs(actual - target) <= TOL if op == "==" else actual >= target - TOL
    if ok:
        return None
    return f"{name}: expected {label} {op} {target}, got {actual:.4f}"


def check(name: str, report: Report) -> list[str]:
    """Verify the harness reports what the definitions require."""
    want = EXPECTED[name]
    problems = [
        assert_metric(name, "leak rate", report.leak_rate, want["leak"]),
        assert_metric(name, "recall/ceiling", report.recall_vs_ceiling, want["ceiling"]),
    ]
    return [p for p in problems if p]


def main() -> int:
    if not GOLDENS.exists():
        print(f"No golden dataset at {GOLDENS}")
        print("Build it with:  uv run python scripts/build_goldens.py")
        return 1

    cases = load(GOLDENS)

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            corpus = load_corpus_index(cur)
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        print("Start it with:  docker compose up -d")
        return 1

    print(f"\n{len(cases)} cases, {len(corpus)} documents, {len(principals)} personas")
    print("=" * 52)

    problems: list[str] = []
    for cls in (NullRetriever, LeakyRetriever, OracleRetriever):
        retriever = cls(corpus)
        report = run(retriever, cases, principals, corpus)
        summarize(retriever.name, report)
        problems.extend(check(retriever.name, report))

    print()
    if problems:
        print("HARNESS MISCALIBRATED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("Harness calibrated: it detects total leakage and recognises a perfect score.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
