"""Tests for the scoring logic.

The subtle cases get the most attention: a metric that is merely wrong is
usually caught quickly, but a metric that is wrong only for the unusual
inputs will quietly mislead every comparison built on it.
"""

from __future__ import annotations

from ragguard.access import Grant, Principal
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import CaseResult, Report, RetrievedDoc, score_case

ENG = Principal("eng@acme.test", "acme", (Grant(clearance="internal"),))
NEWHIRE = Principal("newhire@acme.test", "acme", ())
EXEC = Principal("exec@acme.test", "acme", (Grant(clearance="restricted"),))

PUBLIC = RetrievedDoc("acme://public.md", "acme", "company", "public")
INTERNAL = RetrievedDoc("acme://eng.md", "acme", "engineering", "internal")
SECRET = RetrievedDoc("acme://board.md", "acme", "board-meetings", "restricted")
OTHER_TENANT = RetrievedDoc("globex://x.md", "globex", "engineering", "public")

CORPUS = {d.uri: d for d in (PUBLIC, INTERNAL, SECRET, OTHER_TENANT)}


def case(persona: str, relevant: tuple[str, ...], cls: str = "local") -> GoldenCase:
    return GoldenCase("c1", "acme", persona, "q", cls, relevant)


class TestLeakDetection:
    def test_forbidden_tier_is_a_leak(self):
        result = score_case(case(ENG.email, (SECRET.uri,)), ENG, [SECRET], CORPUS)
        assert result.has_leak
        assert result.leaked == (SECRET.uri,)

    def test_cross_tenant_is_a_leak(self):
        result = score_case(case(EXEC.email, (PUBLIC.uri,)), EXEC, [OTHER_TENANT], CORPUS)
        assert result.has_leak

    def test_permitted_documents_are_not_leaks(self):
        result = score_case(case(ENG.email, (INTERNAL.uri,)), ENG, [PUBLIC, INTERNAL], CORPUS)
        assert not result.has_leak

    def test_retriever_cannot_relabel_its_way_out_of_a_leak(self):
        # A retriever returning the restricted board document, but claiming
        # it is public engineering material. Authorization must be resolved
        # from the corpus, never from what the retriever asserts — otherwise
        # the component under test is grading itself.
        liar = RetrievedDoc(SECRET.uri, "acme", "engineering", "public")
        result = score_case(case(ENG.email, (INTERNAL.uri,)), ENG, [liar], CORPUS)
        assert result.has_leak

    def test_unknown_uri_counts_as_a_leak(self):
        # A document the corpus does not contain is a fabricated citation.
        ghost = RetrievedDoc("acme://does-not-exist.md", "acme", "company", "public")
        result = score_case(case(ENG.email, (INTERNAL.uri,)), ENG, [ghost], CORPUS)
        assert result.has_leak


class TestAllowedRecall:
    def test_none_when_nothing_is_permitted(self):
        # A new hire asking about board minutes should get nothing. That is
        # correct behaviour and must not be scored as 0% recall.
        result = score_case(case(NEWHIRE.email, (SECRET.uri,)), NEWHIRE, [], CORPUS)
        assert result.n_allowed_relevant == 0
        assert result.allowed_recall is None

    def test_counts_only_permitted_relevant_documents(self):
        c = case(ENG.email, (INTERNAL.uri, SECRET.uri))
        result = score_case(c, ENG, [INTERNAL], CORPUS)
        # SECRET is relevant but forbidden, so it is not part of the target.
        assert result.n_allowed_relevant == 1
        assert result.allowed_recall == 1.0

    def test_undeliverable_cases_excluded_from_the_average(self):
        report = Report(results=[
            CaseResult("a", "local", "p", "acme", n_allowed_relevant=2,
                       n_allowed_relevant_found=1),
            CaseResult("b", "local", "p", "acme", n_allowed_relevant=0,
                       n_allowed_relevant_found=0),
        ])
        # Only case "a" counts: 0.5, not (0.5 + 0) / 2.
        assert report.allowed_recall == 0.5


class TestCeiling:
    def test_ceiling_is_one_when_relevant_fits_in_k(self):
        r = CaseResult("a", "global", "p", "acme", n_allowed_relevant=5, k=10)
        assert r.ceiling_recall == 1.0

    def test_ceiling_caps_when_relevant_exceeds_k(self):
        # 25 relevant documents, 10 slots: nothing can score above 40%.
        r = CaseResult("a", "global", "p", "acme", n_allowed_relevant=25, k=10)
        assert r.ceiling_recall == 0.4

    def test_recall_vs_ceiling_credits_a_perfect_retriever(self):
        report = Report(results=[
            CaseResult("a", "global", "p", "acme", n_allowed_relevant=25,
                       n_allowed_relevant_found=10, k=10),
        ])
        assert report.allowed_recall == 0.4      # looks like failure
        assert report.recall_vs_ceiling == 1.0   # is actually perfect


class TestLeakRateIsCaseLevel:
    def test_one_leaked_document_condemns_the_case(self):
        # Deliberately case-level: 1 forbidden document among 99 correct ones
        # is a breach, and a document-level average would report a soothing
        # 1% and bury it.
        report = Report(results=[
            CaseResult("a", "local", "p", "acme", leaked=("x",), n_retrieved=100),
            CaseResult("b", "local", "p", "acme", leaked=(), n_retrieved=100),
        ])
        assert report.leak_rate == 0.5
        assert report.leaked_doc_count == 1


class TestRecallParity:
    def test_parity_is_one_when_privilege_does_not_matter(self):
        report = Report(results=[
            CaseResult("a", "local", "newhire@a", "acme", n_allowed_relevant=2,
                       n_allowed_relevant_found=2),
            CaseResult("b", "local", "exec@a", "acme", n_allowed_relevant=2,
                       n_allowed_relevant_found=2),
        ])
        assert report.recall_parity()["acme"] == 1.0

    def test_parity_exposes_degraded_low_privilege_retrieval(self):
        # The signature post-filter failure: the exec gets a full page, the
        # new hire gets a fraction, and no other metric moves.
        report = Report(results=[
            CaseResult("a", "local", "newhire@a", "acme", n_allowed_relevant=4,
                       n_allowed_relevant_found=1),
            CaseResult("b", "local", "exec@a", "acme", n_allowed_relevant=4,
                       n_allowed_relevant_found=4),
        ])
        assert report.recall_parity()["acme"] == 0.25

    def test_parity_omitted_without_both_ends(self):
        report = Report(results=[
            CaseResult("a", "local", "eng@a", "acme", n_allowed_relevant=1,
                       n_allowed_relevant_found=1),
        ])
        assert "acme" not in report.recall_parity()
