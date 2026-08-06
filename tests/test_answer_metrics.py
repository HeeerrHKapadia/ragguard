"""Tests for answer-level grading: faithfulness and leak detection.

Everything here is a pure function or a small in-memory checker over synthetic
Answers and DocRecords, so no database and no model are needed. These pin the
two properties that make a generated answer trustworthy: every claim is
grounded, and the visible text never exposes forbidden content — whether by
citing it, quoting it, or citing a source that does not exist.
"""

from __future__ import annotations

from ragguard.access import Grant, Principal
from ragguard.eval.answer_metrics import (
    AnswerLeakChecker,
    AnswerLeakResult,
    AnswerReport,
    DocRecord,
    FaithfulnessResult,
    check_faithfulness,
    ngrams,
)
from ragguard.generation.base import Answer, Citation, Claim

NEWHIRE = Principal(
    email="n@gitlab.test",
    tenant_slug="gitlab",
    grants=(Grant(clearance="public"),),
)

PERMITTED_URI = "gitlab/handbook/vacation.md"
FORBIDDEN_URI = "gitlab/board/strategy.md"

PERMITTED_DOC = DocRecord(
    uri=PERMITTED_URI,
    tenant="gitlab",
    section="handbook",
    tier="public",
    text="All team members accrue vacation days each calendar month.",
)
FORBIDDEN_DOC = DocRecord(
    uri=FORBIDDEN_URI,
    tenant="gitlab",
    section="board",
    tier="restricted",
    text="Confidential acquisition negotiations target the rival analytics startup soon.",
)


def _permitted_citation() -> Citation:
    return Citation(marker=1, uri=PERMITTED_URI, title="Vacation", tier="public")


def test_ngrams_shared_and_disjoint():
    text = "the quick brown fox jumps over the lazy dog today"
    assert ngrams(text) & ngrams(text)
    assert not (ngrams(text) & ngrams("completely unrelated wording sits elsewhere without overlap"))


def test_faithfulness_valid_citation_is_faithful():
    answer = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="extractive",
        claims=(Claim(text="Vacation accrues monthly.", citation=1),),
        citations=(_permitted_citation(),),
    )
    faith = check_faithfulness(answer)
    assert faith.faithful is True
    assert faith.grounded_ratio == 1.0


def test_faithfulness_uncited_or_missing_marker_is_unsupported():
    uncited = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="extractive",
        claims=(Claim(text="Floating statement.", citation=None),),
    )
    faith = check_faithfulness(uncited)
    assert faith.faithful is False
    assert faith.unsupported == ("Floating statement.",)

    dangling = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="extractive",
        claims=(Claim(text="Cites a marker with no source.", citation=7),),
        citations=(_permitted_citation(),),
    )
    faith = check_faithfulness(dangling)
    assert faith.faithful is False
    assert faith.total_claims == 1
    assert faith.supported_claims == 0


def test_leak_clean_answer_citing_permitted_doc():
    checker = AnswerLeakChecker([PERMITTED_DOC, FORBIDDEN_DOC])
    answer = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="extractive",
        claims=(Claim(text="Vacation accrues monthly.", citation=1),),
        citations=(_permitted_citation(),),
    )
    result = checker.check(answer, NEWHIRE)
    assert result.has_leak is False


def test_leak_cited_forbidden_doc():
    checker = AnswerLeakChecker([PERMITTED_DOC, FORBIDDEN_DOC])
    answer = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="llm",
        claims=(Claim(text="Board plans an acquisition.", citation=1),),
        citations=(Citation(marker=1, uri=FORBIDDEN_URI, title="Strategy", tier="restricted"),),
    )
    result = checker.check(answer, NEWHIRE)
    assert FORBIDDEN_URI in result.cited_forbidden
    assert result.has_leak is True


def test_leak_content_reproduction_without_citation():
    checker = AnswerLeakChecker([PERMITTED_DOC, FORBIDDEN_DOC])
    # Reproduces >= 6 consecutive distinctive words from the forbidden doc in an
    # uncited claim — the injected-leak pattern the content check exists to catch.
    answer = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="llm",
        claims=(
            Claim(
                text="Confidential acquisition negotiations target the rival "
                "analytics startup soon.",
                citation=None,
            ),
        ),
    )
    result = checker.check(answer, NEWHIRE)
    assert FORBIDDEN_URI in result.content_leak
    assert result.has_leak is True


def test_leak_unknown_citation():
    checker = AnswerLeakChecker([PERMITTED_DOC, FORBIDDEN_DOC])
    answer = Answer(
        query="q",
        persona=NEWHIRE.email,
        backend="llm",
        claims=(Claim(text="From a phantom source.", citation=1),),
        citations=(Citation(marker=1, uri="gitlab/ghost.md", title="Ghost", tier="public"),),
    )
    result = checker.check(answer, NEWHIRE)
    assert "gitlab/ghost.md" in result.unknown_citations
    assert result.has_leak is True


def test_answer_report_leak_rate_aggregates():
    report = AnswerReport()
    clean = AnswerLeakResult()
    leaked = AnswerLeakResult(cited_forbidden=(FORBIDDEN_URI,))
    faith = FaithfulnessResult(total_claims=1, supported_claims=1)

    report.add(clean, faith)
    report.add(leaked, faith)

    assert report.answer_leak_rate == 0.5
