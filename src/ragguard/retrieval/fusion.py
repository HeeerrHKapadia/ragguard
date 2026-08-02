"""Reciprocal Rank Fusion.

Combining two rankers means reconciling scores that are not comparable. A
cosine distance of 0.31 and a full-text rank of 0.084 describe entirely
different quantities on entirely different scales, and normalising them into
agreement requires assumptions about distributions that neither ranker
promises to hold.

RRF sidesteps the problem by discarding the scores and keeping only the
positions. A document's contribution from each ranker is 1/(k + rank), so
what matters is *that* a ranker placed it third, not how confident it
claimed to be. Scale mismatches, non-linear score distributions and one
ranker being systematically overconfident all stop mattering.

The trade is that genuine confidence information is thrown away — a runaway
best match and a narrow win look identical. In practice that loss is
consistently smaller than the damage done by badly normalised scores, which
is why RRF remains the default in production hybrid search.
"""

from __future__ import annotations

# The smoothing constant from the original RRF paper. It flattens the gap
# between the top positions: without it, rank 1 would contribute twice what
# rank 2 does, letting a single ranker's top pick dominate the fusion. At
# k=60 the difference between rank 1 and rank 2 is under two percent.
#
# One property worth knowing, because it is easy to assume the opposite:
# RRF does not reward consensus unconditionally. Since 1/x is convex,
# 1/(k+1) + 1/(k+3) is always greater than 2/(k+2) for any k — so a document
# ranked first by one ranker and third by the other beats one ranked second
# by both. Consensus only wins once the disagreement is wide enough for the
# curve to catch up. Both cases are pinned down in tests/test_fusion.py.
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one ranking, best first.

    Each inner list is one ranker's output in rank order. Ties are broken by
    ID so the output is deterministic — without that, two documents with
    identical fused scores could swap places between runs and quietly make
    the evaluation unreproducible.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
