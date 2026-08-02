"""Dry-run graph extraction and report what the corpus can actually support.

Run before building anything in Neo4j, for the same reason corpus_report.py
runs before seeding: if the graph turns out to be mostly disconnected, that
is worth discovering now rather than after Phase 5 measures a traversal that
had nowhere to go.

The number that matters is link resolution. The corpus is a sample — 300 of
GitLab's 4137 documents — so most links point at pages that were never
ingested. A link only becomes an edge when both ends are present, which
means edge count falls roughly with the square of the sampling rate.

Run:  uv run python scripts/graph_report.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.corpus import build_corpus
from ragguard.graph.extract import extract, resolve_link


def main() -> int:
    _cfg, corpora = build_corpus()

    grand_edges = 0
    grand_concepts = 0
    problems: list[str] = []

    for tenant in sorted(corpora):
        docs = corpora[tenant]
        known = {d.source_uri for d in docs}

        raw_links = 0
        resolved = 0
        edges: set[tuple[str, str]] = set()
        concept_count = Counter()
        docs_with_edges: set[str] = set()

        for doc in docs:
            found = extract(doc.source_uri, doc.text)
            raw_links += len(found.links)

            for target in found.links:
                hit = resolve_link(target, doc.source_uri, known, tenant)
                if hit and hit != doc.source_uri:
                    resolved += 1
                    edges.add((doc.source_uri, hit))
                    docs_with_edges.add(doc.source_uri)
                    docs_with_edges.add(hit)

            for concept in found.concepts:
                concept_count[concept] += 1

        # A concept naming only one document connects nothing to anything.
        shared = {c: n for c, n in concept_count.items() if n >= 2}
        rate = resolved / raw_links if raw_links else 0.0
        connected = len(docs_with_edges) / len(docs) if docs else 0.0

        print(f"\n{tenant}  —  {len(docs)} documents")
        print(f"  raw links            {raw_links:>6}")
        print(f"  resolved into corpus {resolved:>6}  ({rate:.1%})")
        print(f"  distinct edges       {len(edges):>6}")
        print(f"  docs with any edge   {len(docs_with_edges):>6}  ({connected:.1%})")
        print(f"  concepts (headings)  {len(concept_count):>6}")
        print(f"  shared by 2+ docs    {len(shared):>6}")

        grand_edges += len(edges)
        grand_concepts += len(shared)

        if connected < 0.25:
            problems.append(
                f"{tenant}: only {connected:.0%} of documents have a link edge"
            )

    print(f"\n{'=' * 46}")
    print(f"link edges           {grand_edges}")
    print(f"shared concepts      {grand_concepts}")

    if problems:
        print("\nSPARSE:")
        for p in problems:
            print(f"  - {p}")
        print(
            "\n  Link edges alone will not carry multi-hop traversal. Concept\n"
            "  nodes are what connects documents that never linked to each\n"
            "  other, so they are load-bearing here rather than decorative."
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
