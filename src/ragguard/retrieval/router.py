"""Route a query to the retrieval strategy that suits it.

Phase 3 measured that no single configuration is best. Dense retrieval wins
on point lookups; cross-encoder reranking wins by ten points on
cross-document questions and costs 174x throughput. Running reranking on
everything is unaffordable, and running it on nothing gives up the gain.

So the query has to be classified before it is answered — without knowing
the label, which the evaluation dataset has and a real system does not.

Two signals are combined:

1. Query text cues ("compare", "across", "all teams", …) that mark global
   questions without waiting on a probe.
2. The shape of the dense score distribution from a cheap probe. A question
   with one correct document produces a peaked distribution; a question
   whose answer is spread across a section produces a flat one.

When the decision is local, the probe results are reused. When it is global
and the expensive path can rescore an existing shortlist, the probe is
handed over instead of being thrown away — that was pure overhead before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragguard.access import Principal
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc

# Below this spread between the best and worst of the probe results, the
# ranking is considered flat and the query is treated as global. Calibrated
# by scripts/router_benchmark.py against the labelled dataset — a threshold
# chosen by eye would be a guess dressed up as a decision.
FLATNESS_THRESHOLD = 0.055

# How many results to look at when judging peakedness. Too few and one
# outlier decides it; too many and the tail dominates a decision that is
# really about the head.
PROBE_K = 10

# Lexical cues that strongly suggest a cross-document question. Kept as
# word-boundary patterns so "overall" in "overalls" does not trip them.
_GLOBAL_CUES = re.compile(
    r"\b("
    r"compare|comparison|versus|vs\.?|across|difference|differences|"
    r"between .+ and|all (?:teams|departments|groups|sections|policies)|"
    r"how do .+ differ|what (?:are|is) the (?:common|shared)|"
    r"overview of|summar(?:y|ise|ize)|throughout|organisation[- ]wide|"
    r"organization[- ]wide|company[- ]wide"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutingDecision:
    query_class: str
    spread: float
    reason: str = "probe"


def text_suggests_global(query: str) -> bool:
    """Cheap pre-probe cue. Prefer false negatives over false positives —
    wrongly sending a local query to rerank is the expensive mistake."""
    return bool(_GLOBAL_CUES.search(query))


def classify(
    results: list[RetrievedDoc],
    threshold: float = FLATNESS_THRESHOLD,
    query: str = "",
) -> RoutingDecision:
    """Peaked distribution -> local. Flat distribution / text cues -> global.

    Scores here are cosine distances, so smaller is better and the spread is
    last minus first. An empty or single-result probe cannot be flat in any
    meaningful sense, so it falls to local — the cheaper path, which is the
    right way to be wrong — unless the query text itself is clearly global.
    """
    if query and text_suggests_global(query):
        spread = 0.0
        if len(results) >= 2:
            scores = [doc.score for doc in results]
            spread = max(scores) - min(scores)
        return RoutingDecision("global", spread, reason="text")

    if len(results) < 2:
        return RoutingDecision("local", 0.0, reason="probe")

    scores = [doc.score for doc in results]
    spread = max(scores) - min(scores)
    return RoutingDecision(
        "global" if spread < threshold else "local",
        spread,
        reason="probe",
    )


class RoutedRetriever:
    """Cheap path by default, expensive path when the query looks global."""

    name = "routed"

    def __init__(self, local, global_, probe=None, threshold: float = FLATNESS_THRESHOLD):
        self.local = local
        self.global_ = global_
        # The probe defaults to the local retriever, so classification costs
        # nothing extra: its results are reused when the decision is local.
        self.probe = probe or local
        self.threshold = threshold
        self.decisions: list[tuple[str, str]] = []

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        probe_results = self.probe.retrieve(case, principal, max(k, PROBE_K))
        decision = classify(probe_results, self.threshold, query=case.query)
        self.decisions.append((case.query_class, decision.query_class))

        if decision.query_class == "local":
            if self.probe is self.local:
                return probe_results[:k]
            return self.local.retrieve(case, principal, k)

        # Prefer rescoring the probe shortlist over discarding that work and
        # re-running the full expensive stack from scratch.
        rerank = getattr(self.global_, "rerank_candidates", None)
        if callable(rerank) and probe_results:
            return rerank(case, probe_results, k)

        return self.global_.retrieve(case, principal, k)

    def accuracy(self) -> dict[str, float]:
        """How often the router agreed with the dataset's label."""
        if not self.decisions:
            return {}
        correct = sum(1 for actual, predicted in self.decisions if actual == predicted)
        by_class: dict[str, list[int]] = {}
        for actual, predicted in self.decisions:
            by_class.setdefault(actual, []).append(1 if actual == predicted else 0)
        out = {"overall": correct / len(self.decisions)}
        for cls, hits in sorted(by_class.items()):
            out[cls] = sum(hits) / len(hits)
        return out
