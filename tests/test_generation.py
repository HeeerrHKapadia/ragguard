"""Tests for the guarded generators.

These are pure unit tests: they build synthetic Principals and Snippets by
hand and never touch a database, a network, or a real LLM. The point is to
pin down the generation *contract* — every claim cites a permitted source, an
empty result is a neutral refusal, and the LLM path can only ever keep valid
citations — independent of any live backend.
"""

from __future__ import annotations

from ragguard.access import Grant, Principal
from ragguard.eval.answer_metrics import check_faithfulness
from ragguard.generation.base import Snippet
from ragguard.generation.extractive import ExtractiveGenerator
from ragguard.generation.llm import LLMGenerator

NEWHIRE = Principal(
    email="n@gitlab.test",
    tenant_slug="gitlab",
    grants=(Grant(clearance="public"),),
)


def _snippet(marker: str) -> Snippet:
    return Snippet(
        uri=f"gitlab/handbook/{marker}.md",
        title=f"Handbook {marker}",
        tenant="gitlab",
        section="handbook",
        tier="public",
        text=f"The {marker} policy states that all team members must submit reports.",
    )


def test_extractive_two_snippets_yields_two_grounded_claims():
    snippets = [_snippet("alpha"), _snippet("beta")]
    answer = ExtractiveGenerator().generate("what is the policy?", NEWHIRE, snippets)

    assert len(answer.claims) == 2
    markers = {c.marker for c in answer.citations}
    for claim in answer.claims:
        assert claim.citation is not None
        assert claim.citation in markers

    faith = check_faithfulness(answer)
    assert faith.faithful is True
    assert faith.grounded_ratio == 1.0


def test_extractive_empty_snippets_is_a_neutral_refusal():
    answer = ExtractiveGenerator().generate("anything secret?", NEWHIRE, [])

    assert answer.claims == ()
    assert answer.citations == ()
    assert answer.preamble != ""
    # The refusal must not reveal or reference any document: no source list,
    # no uris, and no leaked existence of forbidden material.
    assert "Sources:" not in answer.text
    assert "://" not in answer.text
    assert "gitlab/" not in answer.text


def test_llm_parse_keeps_in_range_markers():
    gen = LLMGenerator(api_key="x")
    gen._complete = lambda system, user: "Alpha statement [1]. Beta statement [2]."
    snippets = [_snippet("alpha"), _snippet("beta")]

    answer = gen.generate("what is the policy?", NEWHIRE, snippets)

    assert len(answer.claims) == 2
    assert [c.citation for c in answer.claims] == [1, 2]
    faith = check_faithfulness(answer)
    assert faith.faithful is True
    assert faith.grounded_ratio == 1.0


def test_llm_parse_drops_out_of_range_marker():
    gen = LLMGenerator(api_key="x")
    gen._complete = lambda system, user: "Gamma [9]."
    snippets = [_snippet("alpha")]

    answer = gen.generate("what is the policy?", NEWHIRE, snippets)

    assert len(answer.claims) == 1
    assert answer.claims[0].citation is None
    faith = check_faithfulness(answer)
    assert faith.faithful is False
    assert answer.claims[0].text in faith.unsupported


def test_from_env_without_key_returns_extractive(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = LLMGenerator.from_env()
    assert isinstance(gen, ExtractiveGenerator)
