"""Policy SQL builders stay aligned with the oracle shape."""

from __future__ import annotations

from ragguard.access import Grant, Principal
from ragguard.retrieval.filters import visibility_sql, visibility_sql_on_chunks


def _principal() -> Principal:
    return Principal(
        email="finance@example",
        tenant_slug="gitlab",
        grants=(Grant(clearance="internal", elevated_sections=("finance",)),),
    )


class TestVisibilitySql:
    def test_document_form_references_document_columns(self):
        sql, params = visibility_sql(_principal())
        assert "d.sensitivity" in sql
        assert "d.section" in sql
        assert params["tenant"] == "gitlab"
        assert params["elev_sections"] == ["finance"]

    def test_chunk_form_references_chunk_columns(self):
        sql, params = visibility_sql_on_chunks(_principal())
        assert "c.sensitivity" in sql
        assert "c.section" in sql
        assert "d.sensitivity" not in sql
        assert params["clearance"] == 1
