"""Corpus discovery: turn handbook markdown files into documents with tiers.

Deliberately does no database work — it is pure functions over the filesystem
so it can be unit-tested without Postgres running. The script that writes to
the database imports from here.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragguard.config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config" / "tenants.yaml"

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    tenant_slug: str
    source_uri: str
    title: str
    section: str
    tier: str
    text: str
    content_hash: str


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def strip_frontmatter(raw: str) -> tuple[dict, str]:
    """Split YAML frontmatter from the markdown body.

    Frontmatter is metadata, not prose. Leaving it in the body would let a
    retriever match on scaffolding like `sidebar: Handbook` — noise that
    appears in thousands of documents and discriminates between none of them.
    """
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, raw[match.end():]


def derive_title(meta: dict, body: str, path: Path) -> str:
    """Best available title: frontmatter, then first H1, then the filename."""
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    heading = HEADING_RE.search(body)
    if heading:
        return heading.group(1).strip()
    # `_index.md` names the directory, not the file.
    stem = path.parent.name if path.stem == "_index" else path.stem
    return stem.replace("-", " ").replace("_", " ").title()


def section_of(rel_path: Path) -> str:
    """First path component — the department the document belongs to."""
    parts = rel_path.parts
    return parts[0] if len(parts) > 1 else "root"


def assign_tier(rel_path: Path, rules: list[dict], default_tier: str) -> str:
    """First matching prefix rule wins, else the tenant default.

    Uses POSIX-style joining so the rules in tenants.yaml stay readable and
    behave identically on Windows, where Path.parts would otherwise render
    separators as backslashes and never match a rule like `departments/legal`.
    """
    rel = rel_path.as_posix()
    for rule in rules:
        prefix = rule["prefix"]
        if rel == prefix or rel.startswith((f"{prefix}/", f"{prefix}.")):
            return rule["tier"]
    return default_tier


def discover(tenant: dict, min_chars: int) -> list[Document]:
    """Walk one tenant's handbook and build Document records."""
    root = PROJECT_ROOT / tenant["root"]
    if not root.exists():
        raise FileNotFoundError(f"corpus missing for {tenant['slug']}: {root}")

    rules = tenant.get("tier_rules", [])
    default_tier = tenant.get("default_tier", "internal")

    # Sort on the POSIX relative path string, not on Path objects.
    #
    # Path comparison is platform-dependent: PureWindowsPath compares
    # case-insensitively while PurePosixPath is case-sensitive, so
    # `README.md` and `about.md` order differently on Windows and Linux.
    # Sampling picks evenly-spaced items from this list, so a different
    # order means a different corpus — reproducible on one machine and
    # not the other, which is the worst kind of reproducible.
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md"), key=lambda p: p.relative_to(root).as_posix()):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = strip_frontmatter(raw)
        body = body.strip()

        # Stub pages (nav placeholders, redirects) add index size and noise
        # without adding retrievable content.
        if len(body) < min_chars:
            continue

        rel = path.relative_to(root)
        docs.append(
            Document(
                tenant_slug=tenant["slug"],
                source_uri=f"{tenant['slug']}://{rel.as_posix()}",
                title=derive_title(meta, body, path),
                section=section_of(rel),
                tier=assign_tier(rel, rules, default_tier),
                text=body,
                content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            )
        )
    return docs


def _spread(items: list[Document], keep: int) -> list[Document]:
    """Keep `keep` items spread evenly across the list.

    Taking the first N would bias toward whatever sorts alphabetically first.
    Evenly spaced picks give a spread of the group instead of a corner of it,
    and because it is index arithmetic rather than randomness, re-running
    yields the identical corpus.
    """
    if keep <= 0:
        return []
    if len(items) <= keep:
        return list(items)
    stride = len(items) / keep
    return [items[int(i * stride)] for i in range(keep)]


def stratified_sample(
    docs: list[Document],
    max_per_section: int,
    max_per_tenant: int,
    never_sample_tiers: set[str],
    cap_protected_sections: bool = False,
) -> list[Document]:
    """Trim a tenant's documents while protecting the tiers that matter.

    Two passes. Cap each section so no department dominates, then trim to the
    tenant budget — skipping the protected tiers, since restricted documents
    are what the leak tests are built on and public handbooks contain few of
    them.

    `cap_protected_sections` decides whether protected tiers also face the
    per-section cap, and it defaults to False because that is what produced
    every published measurement.

    The distinction is invisible while only `restricted` is protected: those
    sections are all smaller than the cap anyway. It stops being invisible
    the moment `public` is protected for a smaller demo build — one tenant's
    corpus grew from 300 documents to 774, because "never sampled away" had
    quietly meant "never capped" and the largest public sections arrived
    whole.

    Turning the flag on is the correct behaviour and costs one document at
    the default settings, which is enough to change the golden dataset. So it
    is opt-in: the deployment sets it, and the measurements do not move.
    """
    if cap_protected_sections:
        by_section: dict[str, list[Document]] = {}
        for doc in docs:
            by_section.setdefault(doc.section, []).append(doc)
        pool: list[Document] = []
        for section in sorted(by_section):
            pool.extend(_spread(by_section[section], max_per_section))
        protected = [d for d in pool if d.tier in never_sample_tiers]
        kept = [d for d in pool if d.tier not in never_sample_tiers]
    else:
        protected = [d for d in docs if d.tier in never_sample_tiers]
        trimmable = [d for d in docs if d.tier not in never_sample_tiers]

        by_section = {}
        for doc in trimmable:
            by_section.setdefault(doc.section, []).append(doc)
        kept = []
        for section in sorted(by_section):
            kept.extend(_spread(by_section[section], max_per_section))

    # When the protected tiers already fill or exceed the budget, nothing
    # trimmable survives. An earlier `if budget > 0` guard skipped trimming
    # entirely in that case, so asking for a *smaller* corpus produced a
    # larger one — cap 40 yielded 647 documents while cap 80 yielded 240.
    # It never fired while only restricted was protected, because 34 is
    # comfortably under 300.
    budget = max_per_tenant - len(protected)
    if len(kept) > budget:
        kept = _spread(sorted(kept, key=lambda d: d.source_uri), budget)

    return sorted(protected + kept, key=lambda d: d.source_uri)


def build_corpus() -> tuple[dict, dict[str, list[Document]]]:
    """Load config and return (config, {tenant_slug: sampled documents}).

    MAX_DOCS_PER_TENANT overrides the configured cap. It exists for the
    deployment image, where embedding the full corpus at build time is the
    dominant cost and a free build has a time limit.

    The override is an environment variable rather than a config edit on
    purpose. Changing tenants.yaml would regenerate the golden dataset,
    break the determinism check in CI, and silently invalidate every number
    in the README — the measurements and the demo have to be able to
    disagree about corpus size without one corrupting the other.

    Restricted documents stay protected from sampling at any cap, so a
    smaller corpus still contains the material the persona contrast depends
    on. A demo where the executive sees nothing an engineer cannot would
    demonstrate the opposite of the point.
    """
    cfg = load_config()
    sampling = cfg.get("sampling", {})
    max_per_section = sampling.get("max_docs_per_section", 25)
    max_per_tenant = int(
        os.getenv("MAX_DOCS_PER_TENANT", sampling.get("max_docs_per_tenant", 300))
    )

    # NEVER_SAMPLE_TIERS is the other half of the deployment override. The
    # binding constraint on shrinking the corpus is not the total — it is
    # that public documents run out, and a new hire who can only read public
    # material then receives almost nothing. Protecting that tier as well
    # costs 57 documents in total and lets the cap fall much further before
    # the demo stops making sense.
    never_sample = set(
        os.getenv(
            "NEVER_SAMPLE_TIERS",
            ",".join(sampling.get("never_sample_tiers", [])),
        ).split(",")
    ) - {""}
    min_chars = sampling.get("min_chars", 400)

    # Only meaningful alongside an expanded protected set, so it is tied to
    # that override rather than exposed as a third knob nobody would set on
    # its own.
    cap_protected = "NEVER_SAMPLE_TIERS" in os.environ

    result: dict[str, list[Document]] = {}
    for tenant in cfg["tenants"]:
        found = discover(tenant, min_chars)
        result[tenant["slug"]] = stratified_sample(
            found, max_per_section, max_per_tenant, never_sample, cap_protected
        )
    return cfg, result
