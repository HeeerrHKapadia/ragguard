"""Tests for fixed and structure-aware chunking."""

from __future__ import annotations

from ragguard.chunking import chunk_fixed, chunk_markdown, chunk_text


class TestFixed:
    def test_empty(self):
        assert chunk_text("") == []

    def test_overlap_keeps_boundary_span(self):
        text = "a" * 100 + "MARKER" + "b" * 100
        chunks = chunk_fixed(text, chunk_chars=120, overlap_chars=40)
        assert any("MARKER" in c.text for c in chunks)
        assert len(chunks) >= 2


class TestMarkdown:
    def test_keeps_heading_with_section(self):
        text = "# Leave\n\nVacation details here.\n\n# Pay\n\nSalary bands here."
        chunks = chunk_markdown(text, chunk_chars=200, overlap_chars=20)
        assert any(c.text.startswith("# Leave") for c in chunks)
        assert any("# Pay" in c.text for c in chunks)

    def test_strategy_dispatch(self):
        text = "# A\n\nbody\n\n# B\n\nmore"
        md = chunk_text(text, strategy="markdown", chunk_chars=80, overlap_chars=10)
        fixed = chunk_text(text, strategy="fixed", chunk_chars=80, overlap_chars=10)
        assert md[0].text.startswith("# A")
        assert fixed  # still produces something

    def test_oversized_section_falls_back(self):
        text = "# Big\n\n" + ("word " * 500)
        chunks = chunk_markdown(text, chunk_chars=100, overlap_chars=20)
        assert len(chunks) > 1
        assert all(len(c.text) <= 100 + 5 for c in chunks)  # strip slack
