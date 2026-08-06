"""Unit tests for router classification and probe reuse."""

from __future__ import annotations

import pytest

from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc
from ragguard.retrieval.router import (
    RoutedRetriever,
    classify,
    text_suggests_global,
)


def _doc(uri: str, score: float) -> RetrievedDoc:
    return RetrievedDoc(uri=uri, tenant="t", section="s", tier="public", score=score)


class TestTextCues:
    def test_compare_is_global(self):
        assert text_suggests_global("compare leave policies across teams")

    def test_point_lookup_is_not(self):
        assert not text_suggests_global("what is the vacation policy?")


class TestClassify:
    def test_peaked_probe_is_local(self):
        decision = classify([_doc("a", 0.1), _doc("b", 0.3), _doc("c", 0.4)])
        assert decision.query_class == "local"
        assert decision.spread == pytest.approx(0.3)

    def test_flat_probe_is_global(self):
        decision = classify([_doc("a", 0.20), _doc("b", 0.22), _doc("c", 0.24)])
        assert decision.query_class == "global"

    def test_text_cue_overrides_peak(self):
        decision = classify(
            [_doc("a", 0.1), _doc("b", 0.5)],
            query="compare engineering and finance onboarding",
        )
        assert decision.query_class == "global"
        assert decision.reason == "text"


class TestProbeReuse:
    def test_global_reuses_probe_via_rerank_candidates(self):
        probe_docs = [_doc("a", 0.1), _doc("b", 0.12)]
        calls: list[str] = []

        class Probe:
            def retrieve(self, case, principal, k):
                calls.append("probe")
                return probe_docs

        class Local:
            def retrieve(self, case, principal, k):
                calls.append("local")
                return probe_docs

        class Global:
            def retrieve(self, case, principal, k):
                calls.append("global-retrieve")
                return [_doc("z", 0.0)]

            def rerank_candidates(self, case, candidates, k):
                calls.append("rerank")
                return candidates[:k]

        routed = RoutedRetriever(local=Local(), global_=Global(), probe=Probe())
        case = GoldenCase(
            case_id="1", tenant="t", persona="p",
            query="compare policies across departments",
            query_class="global", relevant_uris=(),
        )
        out = routed.retrieve(case, principal=None, k=2)  # type: ignore[arg-type]
        assert [d.uri for d in out] == ["a", "b"]
        assert calls == ["probe", "rerank"]
        assert "global-retrieve" not in calls
