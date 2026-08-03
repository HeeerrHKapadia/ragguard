"""Route a query to the retrieval strategy that suits it.

Phase 3 measured that no single configuration is best. Dense retrieval wins
on point lookups; cross-encoder reranking wins by ten points on
cross-document questions and costs 174x throughput. Running reranking on
everything is unaffordable, and running it on nothing gives up the gain.

So the query has to be classified before it is answered — without knowing
the label, which the evaluation dataset has and a real system does not.

The signal is the shape of the dense score distribution. A question with one
correct document produces a peaked distribution: the best match is clearly
better than the tenth. A question whose answer is spread across a section
produces a flat one, because many documents are similarly related and none
dominates. Peakedness is measurable before any expensive stage runs, from
results already fetched.

A router that is right most of the time still beats paying for reranking
every time, so this reports its own accuracy rather than assuming it.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class RoutingDecision:
    query_class: str
    spread: float


def classify(results: list[RetrievedDoc], threshold: float = FLATNESS_THRESHOLD) -> RoutingDecision:
    """Peaked distribution -> local. Flat distribution -> global.

    Scores here are cosine distances, so smaller is better and the spread is
    last minus first. An empty or single-result probe cannot be flat in any
    meaningful sense, so it falls to local — the cheaper path, which is the
    right way to be wrong.
    """
    if len(results) < 2:
        return RoutingDecision("local", 0.0)

    scores = [doc.score for doc in results]
    spread = max(scores) - min(scores)
    return RoutingDecision("global" if spread < threshold else "local", spread)


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
        decision = classify(probe_results, self.threshold)
        self.decisions.append((case.query_class, decision.query_class))

        if decision.query_class == "local" and self.probe is self.local:
            return probe_results[:k]
        chosen = self.local if decision.query_class == "local" else self.global_
        return chosen.retrieve(case, principal, k)

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
