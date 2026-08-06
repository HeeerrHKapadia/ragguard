"""The leaderboard for guarded generation: answer-level leak + faithfulness.

`scripts/evaluate.py` grades *retrievers* on document-level leak and recall.
This is its counterpart one layer up: it grades *generators* on the surface a
user actually reads — the generated answer — using the same corpus oracle
(`access.can_read`) so a backend can never grade its own output.

Two numbers per backend:

  answer_leak_rate    fraction of answers that expose content the persona may
                      not read — by citing a forbidden/unknown document, or by
                      reproducing distinctive wording from one.
  faithfulness_rate   fraction of claims grounded in a cited, provided source.

The extractive backend is the fixed anchor: it composes answers only from
permitted snippets and cites every claim, so it must score 0.0000 leak and
1.0000 faithfulness. If it doesn't, the harness — not the generator — is broken.

Run:  uv run python scripts/leakbench.py --backend extractive --limit 120
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys
import time

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import Principal, load_principals
from ragguard.config import PROJECT_ROOT
from ragguard.corpus import build_corpus
from ragguard.db import connect
from ragguard.eval.answer_metrics import (
    AnswerLeakChecker,
    AnswerReport,
    DocRecord,
    check_faithfulness,
)
from ragguard.eval.dataset import GoldenCase, load
from ragguard.generation import ExtractiveGenerator, Generator, retrieve_context
from ragguard.generation.llm import LLMGenerator

GOLDENS = PROJECT_ROOT / "eval" / "goldens.jsonl"


def build_records() -> list[DocRecord]:
    """Every document's ACL metadata + text, straight from the corpus source.

    This is the oracle the checker grades against, so it is built from the
    filesystem corpus rather than the database — the same truth the golden
    dataset and access policy are derived from.
    """
    _, corpora = build_corpus()
    records: list[DocRecord] = []
    for docs in corpora.values():
        for doc in docs:
            records.append(
                DocRecord(
                    uri=doc.source_uri,
                    tenant=doc.tenant_slug,
                    section=doc.section,
                    tier=doc.tier,
                    text=doc.text,
                )
            )
    return records


def sample_evenly(cases: list[GoldenCase], limit: int) -> list[GoldenCase]:
    """Keep `limit` cases spread across the dataset with a constant stride.

    Taking the first N would grade one tenant's local queries and nothing
    else. Striding across the sorted goldens keeps every tenant, persona and
    query class represented, and being index arithmetic it is reproducible.
    """
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    stride = max(1, len(cases) // limit)
    return cases[::stride][:limit]


def select_backends(backend: str) -> list[tuple[str, Generator]]:
    """Map the --backend flag to labelled generators, in display order."""
    if backend == "extractive":
        return [("extractive", ExtractiveGenerator())]
    if backend == "llm":
        return [("llm", LLMGenerator.from_env())]
    return [
        ("extractive", ExtractiveGenerator()),
        ("llm", LLMGenerator.from_env()),
    ]


def grade_backend(
    generator: Generator,
    contexts: list[tuple[GoldenCase, Principal, list]],
    checker: AnswerLeakChecker,
) -> tuple[AnswerReport, dict[str, AnswerReport], collections.Counter]:
    """Run one generator over the shared contexts and grade every answer."""
    report = AnswerReport()
    by_persona: dict[str, AnswerReport] = collections.defaultdict(AnswerReport)
    cited_tiers: collections.Counter = collections.Counter()

    for case, principal, snippets in contexts:
        answer = generator.generate(case.query, principal, snippets)
        # Grade against the corpus oracle, folding the exact snippets the persona
        # was shown into the permitted baseline so quoting permitted material is
        # never mistaken for a leak (see AnswerLeakChecker.check).
        leak = checker.check(answer, principal, [s.text for s in snippets])
        faith = check_faithfulness(answer)
        report.add(leak, faith)
        by_persona[case.persona].add(leak, faith)
        for citation in answer.citations:
            cited_tiers[citation.tier] += 1

    return report, by_persona, cited_tiers


def print_leaderboard(rows: list[tuple[str, AnswerReport]]) -> None:
    """The headline table: one row per backend."""
    header = (
        f"  {'backend':<12} {'cases':>6} {'answer_leak_rate':>18} "
        f"{'faithfulness_rate':>19} {'leaked_cases':>13}"
    )
    print("\nLeaderboard\n")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, report in rows:
        print(
            f"  {label:<12} {report.n:>6} "
            f"{report.answer_leak_rate:>18.4f} "
            f"{report.faithfulness_rate:>19.4f} "
            f"{report.leaked:>13}"
        )


def print_persona_breakdown(label: str, by_persona: dict[str, AnswerReport]) -> None:
    """Per-persona leak is where a privilege-blind generator gives itself away."""
    print(f"\n  {label}: answer_leak_rate by persona")
    print(f"    {'persona':<32} {'cases':>6} {'leak_rate':>10} {'faithful':>10}")
    print("    " + "-" * 60)
    for persona in sorted(by_persona):
        rep = by_persona[persona]
        print(
            f"    {persona:<32} {rep.n:>6} "
            f"{rep.answer_leak_rate:>10.4f} {rep.faithfulness_rate:>10.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("extractive", "llm", "both"),
        default="extractive",
        help="generator backend(s) to score",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="max graded cases, sampled evenly across the goldens",
    )
    parser.add_argument("--k", type=int, default=5, help="snippets retrieved per query")
    args = parser.parse_args()

    if not GOLDENS.exists():
        print(f"No golden dataset at {GOLDENS}")
        print("Build it with:  uv run python scripts/build_goldens.py")
        return 1

    started = time.time()

    records = build_records()
    checker = AnswerLeakChecker(records)

    everything = load(GOLDENS)
    selected = sample_evenly(everything, args.limit)

    backends = select_backends(args.backend)
    if args.backend in ("llm", "both") and not os.getenv("OPENAI_API_KEY"):
        print(
            "Note: OPENAI_API_KEY not set — the LLM backend fell back to the "
            "extractive generator (LLMGenerator.from_env). Scoring it anyway."
        )

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            if not principals:
                print("Seed the database first: uv run python scripts/seed.py")
                return 1

            print(
                f"\n{len(records)} documents, {len(everything)} goldens "
                f"({len(selected)} sampled), {len(principals)} personas, k={args.k}"
            )

            # Retrieve context once per case and reuse it across backends, so
            # the query is embedded once and every backend is graded on the
            # identical permitted snippets.
            contexts: list[tuple[GoldenCase, Principal, list]] = []
            for case in selected:
                principal = principals.get(case.persona)
                if principal is None:
                    continue
                snippets = retrieve_context(conn, case.query, principal, args.k)
                contexts.append((case, principal, snippets))

            rows: list[tuple[str, AnswerReport]] = []
            details: list[tuple[str, dict[str, AnswerReport], collections.Counter]] = []
            for label, generator in backends:
                report, by_persona, cited_tiers = grade_backend(
                    generator, contexts, checker
                )
                rows.append((label, report))
                details.append((label, by_persona, cited_tiers))
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        print("Start it with:  docker compose up -d")
        return 1

    print_leaderboard(rows)

    for label, by_persona, cited_tiers in details:
        print_persona_breakdown(label, by_persona)
        if cited_tiers:
            tiers = "  ".join(f"{tier}={n}" for tier, n in sorted(cited_tiers.items()))
            print(f"  {label}: cited sources by tier  {tiers}")

    elapsed = time.time() - started
    print(f"\nGraded {len(contexts)} cases per backend in {elapsed:.1f}s")

    # A benchmark failure is a real leak, not a low score. The extractive
    # anchor must read 0.0000; if any backend leaks, surface it as exit 1.
    leaking = [label for label, report in rows if report.answer_leak_rate > 0]
    if leaking:
        print(f"\nBENCHMARK FAILURE: answer leak detected in {', '.join(leaking)}")
        for label, report in rows:
            if report.answer_leak_rate > 0:
                sample = ", ".join(sorted(set(report.leak_uris))[:5])
                print(f"  {label}: {report.leaked} leaked answer(s); e.g. {sample}")
        return 1

    print("\nNo answer-level leaks. The guarded generation bar holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
