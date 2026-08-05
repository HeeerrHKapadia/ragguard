"""Tests for corpus sampling.

These exist because of a bug that shipped: asking for a smaller corpus
produced a larger one. Sampling is quiet code — it returns a plausible list
whatever it does — so the invariants get asserted directly rather than
inferred from downstream numbers.
"""

from __future__ import annotations

from ragguard.corpus import Document, _spread, stratified_sample


def doc(uri: str, section: str, tier: str) -> Document:
    return Document(
        tenant_slug="acme", source_uri=uri, title=uri,
        section=section, tier=tier, text="x" * 500, content_hash=uri,
    )


def corpus(counts: dict[tuple[str, str], int]) -> list[Document]:
    out = []
    for (section, tier), n in counts.items():
        out.extend(doc(f"acme://{section}/{i:03d}.md", section, tier) for i in range(n))
    return out


class TestSpread:
    def test_keeps_everything_when_under_budget(self):
        items = corpus({("eng", "internal"): 3})
        assert len(_spread(items, 10)) == 3

    def test_zero_budget_keeps_nothing(self):
        # The original returned everything for keep<=0, which is how a
        # smaller cap produced a bigger corpus.
        items = corpus({("eng", "internal"): 50})
        assert _spread(items, 0) == []

    def test_negative_budget_keeps_nothing(self):
        items = corpus({("eng", "internal"): 50})
        assert _spread(items, -20) == []

    def test_picks_are_spread_not_a_prefix(self):
        items = corpus({("eng", "internal"): 100})
        kept = _spread(items, 5)
        assert len(kept) == 5
        # Taking the first five would bias toward whatever sorts first.
        assert kept[-1] != items[4]

    def test_deterministic(self):
        items = corpus({("eng", "internal"): 100})
        assert [d.source_uri for d in _spread(items, 7)] == \
               [d.source_uri for d in _spread(items, 7)]


class TestStratifiedSample:
    def test_smaller_cap_never_yields_more_documents(self):
        """The invariant the shipped bug violated.

        With public protected and a low cap, protected exceeded the budget,
        trimming was skipped entirely, and cap 40 returned more than cap 80.
        """
        docs = corpus({
            ("company", "public"): 40,
            ("eng", "internal"): 200,
            ("finance", "confidential"): 30,
            ("board", "restricted"): 20,
        })
        sizes = [
            len(stratified_sample(docs, 25, cap, {"restricted", "public"}, True))
            for cap in (40, 60, 80, 120, 200)
        ]
        assert sizes == sorted(sizes), sizes

    def test_protected_tiers_survive_a_tight_budget(self):
        docs = corpus({
            ("company", "public"): 10,
            ("eng", "internal"): 500,
            ("board", "restricted"): 15,
        })
        kept = stratified_sample(docs, 25, 30, {"restricted", "public"}, True)
        tiers = [d.tier for d in kept]
        assert tiers.count("restricted") == 15
        assert tiers.count("public") == 10

    def test_protected_tiers_are_section_capped_when_asked(self):
        docs = corpus({("company", "public"): 200, ("eng", "internal"): 50})
        kept = stratified_sample(docs, 10, 500, {"public"}, True)
        assert sum(1 for d in kept if d.tier == "public") == 10

    def test_protected_tiers_bypass_the_section_cap_by_default(self):
        # The behaviour every published measurement was produced with.
        docs = corpus({("company", "public"): 200, ("eng", "internal"): 50})
        kept = stratified_sample(docs, 10, 500, {"public"}, False)
        assert sum(1 for d in kept if d.tier == "public") == 200

    def test_output_is_sorted_and_deterministic(self):
        docs = corpus({("eng", "internal"): 80, ("board", "restricted"): 5})
        first = stratified_sample(docs, 25, 40, {"restricted"})
        second = stratified_sample(docs, 25, 40, {"restricted"})
        uris = [d.source_uri for d in first]
        assert uris == sorted(uris)
        assert uris == [d.source_uri for d in second]
