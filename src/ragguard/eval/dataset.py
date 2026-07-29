"""Golden dataset: the questions the system gets graded on.

Two kinds of ground truth are needed, and they come from very different places.

*Security* truth is free and exact. Whether a persona may read a document is
decided by access.py, so no labelling is required and there is no ambiguity —
a leak is a leak. This is the unusual luxury of a permission-aware system and
it is why the security metrics can be a hard CI gate rather than a guideline.

*Relevance* truth is the expensive kind. Knowing which documents answer a
question normally means human annotation. The standard workaround is to invert
the problem: instead of writing a question and hunting for its answer, take a
document and derive a question it answers. The source document is then known
to be relevant by construction.

Queries are split into two classes because the whole GraphRAG thesis rests on
the distinction:

  local   answerable from one document. Plain hybrid search should win here.
  global  requires connecting several documents. This is the only place a
          knowledge graph can justify its 3-10x token cost, and without this
          split there is no way to show that it does.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ragguard.corpus import Document

# Titles that identify a page's position in a hierarchy rather than its
# subject. Deriving a query from these produces something that matches
# hundreds of documents equally well, which teaches the eval nothing.
GENERIC_TITLES = {
    "index", "readme", "home", "overview", "handbook", "contents",
    "introduction", "getting started", "about",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by",
    "with", "our", "we", "how", "what", "is", "are", "this", "that", "its",
}

WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class GoldenCase:
    """One graded question.

    Deliberately does NOT store the forbidden document set. That set is
    hundreds of URIs per case and, more importantly, it is derived — storing
    it would let the dataset drift out of sync with the policy it is meant to
    be testing. The scorer recomputes it from access.py at grading time, so
    the policy always has exactly one source of truth.
    """

    case_id: str
    tenant: str
    persona: str
    query: str
    query_class: str          # "local" | "global"
    relevant_uris: tuple[str, ...]


def normalize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2]


def is_generic(title: str) -> bool:
    return title.strip().lower() in GENERIC_TITLES


def local_query(doc: Document) -> str | None:
    """Derive a point-lookup query from a single document's title.

    Title-derived queries are weak supervision, not a substitute for real
    user questions — they share vocabulary with the document, which flatters
    lexical search. That bias is acceptable here because every retriever is
    measured against the same queries, so comparisons stay fair even if the
    absolute numbers run optimistic. Phase 4 replaces these with LLM-written
    questions once there is an API budget in play.
    """
    if is_generic(doc.title):
        return None
    words = normalize(doc.title)
    if len(words) < 2:
        return None
    return " ".join(words)


def global_query(section: str, docs: list[Document]) -> str | None:
    """Derive a cross-document query from a section's shared vocabulary.

    A genuine global question spans documents, so its relevant set is the
    section rather than any single page. Built from terms that recur across
    the section's titles: a word appearing in many titles names the theme,
    while a word appearing in one names a page.
    """
    counts = Counter()
    for doc in docs:
        counts.update(set(normalize(doc.title)))

    # Sort by (-count, word) rather than using Counter.most_common().
    #
    # most_common breaks ties by insertion order, and insertion order here
    # comes from iterating a set. Python randomises string hashes per process
    # by default, so set iteration order — and therefore the tie-break, and
    # therefore the generated query text — changed between runs. Adding the
    # word itself as a secondary sort key removes the ambiguity entirely.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shared = [word for word, n in ranked if n >= 2][:6]
    if len(shared) < 2:
        return None

    topic = " ".join(shared[:4])
    section_words = " ".join(normalize(section.replace("-", " ").replace("/", " ")))
    return f"{section_words} {topic}".strip()


def build_cases(
    corpora: dict[str, list[Document]],
    personas_by_tenant: dict[str, list[str]],
    max_local_per_tenant: int = 60,
    min_section_size: int = 5,
) -> list[GoldenCase]:
    """Generate the full graded question set.

    Every query is asked by every persona in its tenant. That is the point:
    the same question from a new hire and from an exec must return different
    documents, and comparing those two answers is what recall parity measures.
    """
    cases: list[GoldenCase] = []

    for tenant in sorted(corpora):
        docs = corpora[tenant]
        personas = personas_by_tenant.get(tenant, [])
        if not personas:
            continue

        # --- local: one document, one question ---------------------------
        candidates = []
        for doc in docs:
            query = local_query(doc)
            if query:
                candidates.append((query, doc))

        # Spread the picks across the sorted corpus rather than taking the
        # first N, so the sample covers every section and tier.
        if len(candidates) > max_local_per_tenant:
            stride = len(candidates) / max_local_per_tenant
            candidates = [candidates[int(i * stride)] for i in range(max_local_per_tenant)]

        for index, (query, doc) in enumerate(candidates):
            for persona in personas:
                cases.append(
                    GoldenCase(
                        case_id=f"{tenant}-local-{index:04d}-{persona.split('@')[0]}",
                        tenant=tenant,
                        persona=persona,
                        query=query,
                        query_class="local",
                        relevant_uris=(doc.source_uri,),
                    )
                )

        # --- global: a section, one question ------------------------------
        by_section: dict[str, list[Document]] = {}
        for doc in docs:
            by_section.setdefault(doc.section, []).append(doc)

        index = 0
        for section in sorted(by_section):
            group = by_section[section]
            if len(group) < min_section_size:
                continue
            query = global_query(section, group)
            if not query:
                continue
            uris = tuple(sorted(d.source_uri for d in group))
            for persona in personas:
                cases.append(
                    GoldenCase(
                        case_id=f"{tenant}-global-{index:04d}-{persona.split('@')[0]}",
                        tenant=tenant,
                        persona=persona,
                        query=query,
                        query_class="global",
                        relevant_uris=uris,
                    )
                )
            index += 1

    return cases


def save(cases: list[GoldenCase], path: Path) -> None:
    """Write as JSONL — one case per line, diffable in git.

    newline="\\n" is load-bearing. Python otherwise translates line endings
    per platform, so this file would be byte-identical only among people on
    the same OS — and CI checks that regenerating it reproduces the committed
    bytes exactly. Sorted keys for the same reason: dict order must not leak
    into the output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(asdict(case), sort_keys=True) + "\n")


def load(path: Path) -> list[GoldenCase]:
    cases = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                raw = json.loads(line)
                raw["relevant_uris"] = tuple(raw["relevant_uris"])
                cases.append(GoldenCase(**raw))
    return cases
