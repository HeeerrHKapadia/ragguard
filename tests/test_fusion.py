"""Tests for reciprocal rank fusion.

Pure function over plain lists, so no database and no model needed. Fusion
is the kind of code that produces plausible-looking output when subtly
wrong, which is exactly why it gets tested directly rather than only through
end-to-end recall numbers.
"""

from __future__ import annotations

import pytest

from ragguard.retrieval.fusion import reciprocal_rank_fusion
from ragguard.retrieval.hybrid import dedupe, to_websearch_or


class TestFusion:
    def test_unanimous_order_is_preserved(self):
        both = [["a", "b", "c"], ["a", "b", "c"]]
        assert [doc for doc, _ in reciprocal_rank_fusion(both)] == ["a", "b", "c"]

    def test_symmetric_disagreement_favours_the_extremes(self):
        # Counter-intuitive and worth pinning down. "b" is second in both
        # rankers; "a" and "c" are each one ranker's first and the other's
        # third. Consensus does NOT win here:
        #
        #   b = 2/62            = 0.032258
        #   a = 1/61 + 1/63     = 0.032266
        #
        # 1/x is convex, so 1/(k+1) + 1/(k+3) always exceeds 2/(k+2) for any
        # k. Being loved by one ranker beats being liked by both when the
        # positions are symmetric. RRF rewards consensus only once the
        # disagreement is wide enough to overcome that.
        fused = [doc for doc, _ in reciprocal_rank_fusion([
            ["a", "b", "c"],
            ["c", "b", "a"],
        ])]
        assert fused[-1] == "b"

    def test_consensus_wins_once_disagreement_is_wide(self):
        # Same shape, wider spread: "a" is first for one ranker and tenth for
        # the other, while "b" is second in both. Now consensus prevails.
        #
        #   a = 1/61 + 1/70 = 0.030679
        #   b = 2/62        = 0.032258
        fused = [doc for doc, _ in reciprocal_rank_fusion([
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            ["j", "b", "i", "h", "g", "f", "e", "d", "c", "a"],
        ])]
        assert fused[0] == "b"

    def test_documents_missing_from_one_ranker_still_score(self):
        fused = dict(reciprocal_rank_fusion([["a", "b"], ["c"]]))
        assert set(fused) == {"a", "b", "c"}
        assert fused["a"] > fused["b"]

    def test_ties_break_deterministically(self):
        # Symmetric input: "x" and "y" earn identical scores. Sorting by id
        # keeps the output stable across runs — otherwise dict ordering
        # could silently make the evaluation unreproducible.
        first = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
        second = reciprocal_rank_fusion([["y", "x"], ["x", "y"]])
        assert [d for d, _ in first] == [d for d, _ in second]

    def test_weights_shift_the_balance(self):
        rankings = [["a", "b"], ["b", "a"]]
        dense_heavy = reciprocal_rank_fusion(rankings, weights=[3.0, 1.0])
        lexical_heavy = reciprocal_rank_fusion(rankings, weights=[1.0, 3.0])
        assert dense_heavy[0][0] == "a"
        assert lexical_heavy[0][0] == "b"

    def test_mismatched_weights_are_rejected(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_one_empty_ranker_does_not_break_fusion(self):
        # Lexical search returns nothing when no term matches. The dense
        # ranking must survive that intact.
        fused = [doc for doc, _ in reciprocal_rank_fusion([["a", "b"], []])]
        assert fused == ["a", "b"]


class TestQueryTranslation:
    def test_terms_are_or_joined(self):
        # Postgres treats space as AND, which would demand every term appear
        # in the same chunk and return nothing for most real questions.
        assert to_websearch_or("vacation policy") == "vacation OR policy"

    def test_punctuation_is_stripped(self):
        assert to_websearch_or("what's the FY2024 plan?") == "what OR s OR the OR fy2024 OR plan"

    def test_empty_query(self):
        assert to_websearch_or("!!!") == ""


class TestDedupe:
    def test_first_occurrence_wins(self):
        assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert dedupe([]) == []
