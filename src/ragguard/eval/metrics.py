"""Scoring: turn retrieval results into the four numbers that matter.

Definitions are written out in full because a metric whose definition is
fuzzy is a metric that can be quietly gamed — including by accident, by the
person who wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragguard.access import Principal, can_read
from ragguard.eval.dataset import GoldenCase


@dataclass(frozen=True)
class RetrievedDoc:
    """One result. Tier and section come from the corpus, not the retriever —
    a retriever must never get to assert its own authorization metadata."""

    uri: str
    tenant: str
    section: str
    tier: str
    score: float = 0.0


@dataclass
class CaseResult:
    case_id: str
    query_class: str
    persona: str
    tenant: str
    leaked: tuple[str, ...] = ()
    n_retrieved: int = 0
    n_allowed_relevant: int = 0
    n_allowed_relevant_found: int = 0
    k: int = 10

    @property
    def has_leak(self) -> bool:
        return bool(self.leaked)

    @property
    def ceiling_recall(self) -> float | None:
        """The best recall physically achievable at this k.

        A global query may have 25 relevant documents while k is 10, in which
        case even a perfect retriever tops out at 40%. Without tracking this,
        the ceiling is invisible and gets mistaken for a retrieval failure —
        and worse, any attempt to "fix" it tunes the system against a limit
        that is really just the cutoff.
        """
        if self.n_allowed_relevant == 0:
            return None
        return min(self.k, self.n_allowed_relevant) / self.n_allowed_relevant

    @property
    def allowed_recall(self) -> float | None:
        """Fraction of entitled-and-relevant documents actually returned.

        None when the persona is entitled to none of the relevant documents.
        That case is not a failure and must not be averaged in as a zero —
        a new hire asking about board minutes *should* get nothing, and
        scoring that as 0% recall would punish correct behaviour.
        """
        if self.n_allowed_relevant == 0:
            return None
        return self.n_allowed_relevant_found / self.n_allowed_relevant


def score_case(
    case: GoldenCase,
    principal: Principal,
    retrieved: list[RetrievedDoc],
    doc_index: dict[str, RetrievedDoc],
    k: int = 10,
) -> CaseResult:
    """Grade one (persona, query) pair against the policy.

    `doc_index` maps every URI in the corpus to its true tenant/section/tier,
    so relevance and authorization are both resolved against the corpus
    rather than against whatever the retriever claims.
    """
    # Resolve every returned URI against the corpus index, never against the
    # metadata the retriever attached to it. A retriever that mislabels a
    # restricted document as public would otherwise have its own leak waved
    # through — the component under test would be grading itself.
    #
    # A URI absent from the index is counted as a leak rather than skipped:
    # returning a document the corpus does not contain is its own failure,
    # and silently ignoring it is how a fabricated citation slips past.
    leaked = tuple(
        doc.uri for doc in retrieved
        if (truth := doc_index.get(doc.uri)) is None
        or not can_read(principal, truth.tenant, truth.section, truth.tier)
    )

    allowed_relevant = {
        uri for uri in case.relevant_uris
        if (d := doc_index.get(uri)) and can_read(principal, d.tenant, d.section, d.tier)
    }
    returned = {doc.uri for doc in retrieved}

    return CaseResult(
        case_id=case.case_id,
        query_class=case.query_class,
        persona=case.persona,
        tenant=case.tenant,
        leaked=leaked,
        n_retrieved=len(retrieved),
        n_allowed_relevant=len(allowed_relevant),
        n_allowed_relevant_found=len(allowed_relevant & returned),
        k=k,
    )


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)

    # -- security -------------------------------------------------------

    @property
    def leak_rate(self) -> float:
        """Fraction of cases where any forbidden document was returned.

        Case-level rather than document-level, and deliberately so. A system
        that returns one forbidden document in a hundred results has still
        leaked; averaging that against the ninety-nine correct results would
        report a reassuring 1% and hide a breach. This is the CI gate, and
        the only acceptable value is 0.
        """
        if not self.results:
            return 0.0
        return sum(r.has_leak for r in self.results) / len(self.results)

    @property
    def leaked_doc_count(self) -> int:
        """Total forbidden documents returned — the blast radius."""
        return sum(len(r.leaked) for r in self.results)

    # -- utility --------------------------------------------------------

    def _recalls(self, predicate=lambda r: True) -> list[float]:
        return [
            r.allowed_recall for r in self.results
            if predicate(r) and r.allowed_recall is not None
        ]

    @property
    def allowed_recall(self) -> float:
        values = self._recalls()
        return sum(values) / len(values) if values else 0.0

    @property
    def ceiling_recall(self) -> float:
        """Best recall any retriever could achieve at this k."""
        values = [
            r.ceiling_recall for r in self.results if r.ceiling_recall is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def recall_vs_ceiling(self) -> float:
        """Achieved recall as a fraction of what was possible.

        This is the number to compare retrievers on. Raw recall punishes a
        system for a cutoff it did not choose; this does not, so a change in
        it reflects an actual change in retrieval quality.
        """
        ceiling = self.ceiling_recall
        return self.allowed_recall / ceiling if ceiling > 0 else 0.0

    @property
    def over_block_rate(self) -> float:
        """Entitled material the user did not receive.

        Honest caveat: this conflates two causes. A document can be missing
        because a permission filter removed it, or simply because ranking
        put it below the cutoff. From outside the system those look the same.
        Phase 2 separates them by running the identical retriever with and
        without filtering and attributing the difference to the filter.
        """
        return 1.0 - self.allowed_recall

    def recall_by_class(self) -> dict[str, float]:
        out = {}
        for cls in ("local", "global"):
            values = self._recalls(lambda r, c=cls: r.query_class == c)
            out[cls] = sum(values) / len(values) if values else 0.0
        return out

    def recall_by_persona(self) -> dict[str, float]:
        by_persona: dict[str, list[float]] = {}
        for r in self.results:
            if r.allowed_recall is not None:
                by_persona.setdefault(r.persona, []).append(r.allowed_recall)
        return {p: sum(v) / len(v) for p, v in sorted(by_persona.items())}

    # -- the metric this project exists to expose -----------------------

    def recall_parity(self) -> dict[str, float]:
        """Lowest-privilege recall divided by highest-privilege recall, per tenant.

        The failure this catches is specific and silent. Retrieve top-k, then
        drop what the user may not see, and a low-privilege persona ends up
        with a handful of results while an exec gets a full page. Both look
        like the system "worked". Leak rate stays at zero, faithfulness stays
        high, and nothing in a conventional RAG scorecard moves — but the new
        hire is quietly getting a much worse product.

        A value near 1.0 means privilege level does not degrade quality.
        Below ~0.8 means it does.
        """
        by_tenant: dict[str, dict[str, list[float]]] = {}
        for r in self.results:
            if r.allowed_recall is not None:
                by_tenant.setdefault(r.tenant, {}).setdefault(r.persona, []).append(
                    r.allowed_recall
                )

        parity = {}
        for tenant, personas in sorted(by_tenant.items()):
            means = {p: sum(v) / len(v) for p, v in personas.items()}
            # Personas are named by privilege in the seed config; "newhire"
            # is the least privileged and "exec" the most.
            low = next((m for p, m in means.items() if p.startswith("newhire")), None)
            high = next((m for p, m in means.items() if p.startswith("exec")), None)
            if low is not None and high is not None and high > 0:
                parity[tenant] = low / high
        return parity
