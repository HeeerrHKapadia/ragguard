"""Tests for the access policy.

These need no database — access.py is pure functions over plain data, which
is exactly why it was written that way. The policy is the part that must not
silently change, so it gets the tightest tests in the project.
"""

from __future__ import annotations

import pytest

from ragguard.access import Grant, Principal, can_read, section_matches

ENG = Principal(
    email="eng@acme.test",
    tenant_slug="acme",
    grants=(Grant(clearance="internal"),),
)
FINANCE = Principal(
    email="finance@acme.test",
    tenant_slug="acme",
    grants=(Grant(clearance="internal", elevated_sections=("finance",)),),
)
EXEC = Principal(
    email="exec@acme.test",
    tenant_slug="acme",
    grants=(Grant(clearance="restricted"),),
)
NEWHIRE = Principal(email="new@acme.test", tenant_slug="acme", grants=())


class TestTenantIsolation:
    """Tenant boundaries are absolute — no clearance level crosses them."""

    @pytest.mark.parametrize("principal", [NEWHIRE, ENG, FINANCE, EXEC])
    def test_nobody_reads_another_tenant(self, principal):
        assert not can_read(principal, "globex", "engineering", "public")

    def test_exec_is_not_privileged_elsewhere(self):
        # The highest clearance in one tenant confers nothing in another.
        assert can_read(EXEC, "acme", "ceo", "restricted")
        assert not can_read(EXEC, "globex", "ceo", "restricted")


class TestClearanceLadder:
    def test_clearance_implies_everything_below(self):
        assert can_read(EXEC, "acme", "engineering", "public")
        assert can_read(EXEC, "acme", "engineering", "internal")
        assert can_read(EXEC, "acme", "finance", "confidential")
        assert can_read(EXEC, "acme", "ceo", "restricted")

    def test_internal_stops_below_confidential(self):
        assert can_read(ENG, "acme", "engineering", "internal")
        assert not can_read(ENG, "acme", "finance", "confidential")
        assert not can_read(ENG, "acme", "ceo", "restricted")

    def test_ungrouped_user_gets_public_only(self):
        assert can_read(NEWHIRE, "acme", "company", "public")
        assert not can_read(NEWHIRE, "acme", "engineering", "internal")


class TestSectionElevation:
    def test_elevation_grants_inside_its_section(self):
        assert can_read(FINANCE, "acme", "finance", "confidential")

    def test_elevation_does_not_leak_to_other_sections(self):
        # The whole point: finance reads confidential finance material and
        # nothing else confidential.
        assert not can_read(FINANCE, "acme", "legal", "confidential")
        assert not can_read(FINANCE, "acme", "people-group", "confidential")

    def test_elevation_lifts_one_tier_not_all_the_way(self):
        # internal + 1 == confidential, which must not reach restricted.
        assert not can_read(FINANCE, "acme", "finance", "restricted")

    def test_elevation_covers_nested_sections(self):
        assert can_read(FINANCE, "acme", "finance/payroll", "confidential")

    def test_elevation_respects_path_boundaries(self):
        # `finance` must not grant access to a different section that merely
        # starts with the same characters.
        assert not can_read(FINANCE, "acme", "finance-committee", "confidential")


class TestSectionMatching:
    @pytest.mark.parametrize(
        ("section", "elevated", "expected"),
        [
            ("people", "people", True),
            ("people/compensation", "people", True),
            ("people-analytics", "people", False),
            ("peoplesoft", "people", False),
            ("finance", "people", False),
        ],
    )
    def test_prefix_boundary(self, section, elevated, expected):
        assert section_matches(section, elevated) is expected


class TestMultipleGrants:
    def test_grants_union(self):
        multi = Principal(
            email="both@acme.test",
            tenant_slug="acme",
            grants=(
                Grant(clearance="internal", elevated_sections=("finance",)),
                Grant(clearance="internal", elevated_sections=("legal",)),
            ),
        )
        assert can_read(multi, "acme", "finance", "confidential")
        assert can_read(multi, "acme", "legal", "confidential")
        assert not can_read(multi, "acme", "people-group", "confidential")
