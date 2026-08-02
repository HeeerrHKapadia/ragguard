"""Pull structure out of markdown, without an LLM.

Handbooks already contain an authored graph. Writers link one page to
another because the pages are genuinely related, and they write headings
because those name the concepts the page is about. Both are free, exact, and
carry no extraction error — a link either exists or it does not.

That is worth exhausting before paying an LLM to infer relationships that
were explicitly written down.

The limitation is real and worth naming: this recovers relationships between
*documents*, plus the concepts those documents name. It does not recover
entity-to-entity relationships of the "Person A reports to Person B" kind,
which is what LLM extraction adds. The extractor interface is deliberately
narrow so an LLM implementation can be dropped in and the difference
measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# [label](target) — the target group stops at the first closing paren, which
# is wrong for links containing nested parens and right for everything else.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)")

# ATX headings only. Setext headings (underlined with === or ---) are rare in
# these corpora and would collide with the frontmatter delimiter.
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*$", re.MULTILINE)

# Markdown and HTML leftovers that would otherwise become part of a concept
# name: bold markers, inline code, links inside headings, stray tags.
HEADING_CLEAN_RE = re.compile(r"[*_`]|<[^>]+>|\[|\]\([^)]*\)")

STOP_HEADINGS = {
    "overview", "introduction", "summary", "background", "notes", "faq",
    "resources", "links", "references", "see also", "contents", "index",
    "getting started", "how to use this page", "further reading",
}


@dataclass(frozen=True)
class Extraction:
    source_uri: str
    links: tuple[str, ...]      # raw link targets, unresolved
    concepts: tuple[str, ...]   # normalized heading text


def find_links(text: str) -> list[str]:
    """Every link target in the document, in order of appearance."""
    out = []
    for target in LINK_RE.findall(text):
        target = target.strip()
        # External links, anchors and mail links describe nothing about the
        # relationship between two handbook pages.
        if target.startswith(("http://", "https://", "#", "mailto:", "tel:")):
            continue
        out.append(target)
    return out


def normalize_concept(raw: str) -> str:
    cleaned = HEADING_CLEAN_RE.sub("", raw).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def find_concepts(text: str, min_len: int = 3, max_words: int = 6) -> list[str]:
    """Headings, treated as the concepts a document names.

    Long headings are usually sentences rather than concepts, and generic
    ones like "Overview" appear on hundreds of pages, so they connect
    everything to everything and carry no information.
    """
    out: list[str] = []
    seen: set[str] = set()
    for _hashes, raw in HEADING_RE.findall(text):
        name = normalize_concept(raw)
        if len(name) < min_len or len(name.split()) > max_words:
            continue
        if name in STOP_HEADINGS or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def resolve_link(target: str, from_uri: str, known: set[str], tenant: str) -> str | None:
    """Map a raw markdown link onto a document URI in the corpus.

    Handbooks link the way they are published, not the way they are stored:
    GitLab writes `/handbook/values/` for a file at `values/_index.md`, and
    Sourcegraph writes `../people/onboarding`. Both have to be walked back to
    a repository path, then checked against what was actually ingested.

    Returns None when the target is outside the corpus — either a page that
    exists upstream but was not sampled, or a link that was already broken.
    Both are common and neither is an error worth stopping for.
    """
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not target:
        return None

    if target.startswith("/"):
        path = target.lstrip("/")
        # GitLab's published URLs carry a /handbook prefix its repository
        # paths do not.
        for prefix in ("handbook/", "contents/handbook/"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
    else:
        # Relative to the linking document's directory.
        base = from_uri.split("://", 1)[1]
        parts = base.split("/")[:-1]
        for piece in target.split("/"):
            if piece in ("", "."):
                continue
            if piece == "..":
                if parts:
                    parts.pop()
            else:
                parts.append(piece)
        path = "/".join(parts)

    path = path.rstrip("/")
    if not path:
        return None

    # A published path can correspond to several stored filenames.
    for candidate in (
        f"{tenant}://{path}",
        f"{tenant}://{path}.md",
        f"{tenant}://{path}/_index.md",
        f"{tenant}://{path}/index.md",
        f"{tenant}://{path}.mdx",
    ):
        if candidate in known:
            return candidate
    return None


def extract(source_uri: str, text: str) -> Extraction:
    return Extraction(
        source_uri=source_uri,
        links=tuple(find_links(text)),
        concepts=tuple(find_concepts(text)),
    )
