"""Access resolution: can a given user read a given document?

This module is the reference implementation of the policy. Nothing here
touches retrieval — it is deliberately the slow, obvious, readable version.

That separation is the point. Later phases push enforcement down into SQL
filters and an authorization service for speed, and the only way to know
those fast paths are correct is to have a slow path that is obviously
correct to compare them against. When the eval harness reports a leak, this
is the oracle that says what should have happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tiers are totally ordered: clearance at one level implies everything below.
TIER_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

# Every authenticated member of a tenant can read that tenant's public
# material even with no group membership at all. This gives us a genuine
# lowest-privilege persona — useful precisely because it is the one most
# likely to suffer silent recall collapse once filtering is added.
BASE_CLEARANCE = "public"


@dataclass(frozen=True)
class Grant:
    """One group's authority: a baseline tier plus per-section elevation."""

    clearance: str = BASE_CLEARANCE
    elevated_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class Principal:
    """A user reduced to just what matters for an access decision."""

    email: str
    tenant_slug: str
    grants: tuple[Grant, ...] = field(default_factory=tuple)

    @property
    def max_clearance(self) -> str:
        best = BASE_CLEARANCE
        for grant in self.grants:
            if TIER_RANK[grant.clearance] > TIER_RANK[best]:
                best = grant.clearance
        return best


def section_matches(doc_section: str, elevated: str) -> bool:
    """Elevation on `people` also covers `people/compensation`.

    Sections are stored as the document's top-level directory, but elevation
    rules may name a deeper path. Matching on a path-prefix boundary — rather
    than a bare startswith — stops `people` from accidentally granting access
    to a section called `people-analytics`.
    """
    return doc_section == elevated or doc_section.startswith(f"{elevated}/")


def can_read(principal: Principal, doc_tenant: str, doc_section: str, doc_tier: str) -> bool:
    """The whole policy, in one function.

    Order matters. Tenant isolation is checked first and is absolute: no
    clearance level, however high, grants access across a tenant boundary.
    An exec at GitLab is not slightly-authorized to read PostHog's material,
    they are entirely unauthorized, and collapsing that into the same
    comparison as tier ranking is how cross-tenant leaks get written.
    """
    if principal.tenant_slug != doc_tenant:
        return False

    needed = TIER_RANK[doc_tier]

    if TIER_RANK[principal.max_clearance] >= needed:
        return True

    # Not cleared globally — but a group may be elevated in this section.
    # Elevation lifts a group one tier above its baseline within its own
    # section, which is what "the finance team can read confidential finance
    # material" actually means in practice.
    for grant in principal.grants:
        for elevated in grant.elevated_sections:
            if section_matches(doc_section, elevated) and TIER_RANK[grant.clearance] + 1 >= needed:
                return True

    return False


def load_principals(cur) -> dict[str, Principal]:
    """Read every seeded user and their effective grants from the database."""
    cur.execute(
        """SELECT u.email,
                  t.slug,
                  COALESCE(g.clearance, %s),
                  COALESCE(g.elevated_sections, '{}')
             FROM users u
             JOIN tenants t     ON t.id = u.tenant_id
        LEFT JOIN user_groups ug ON ug.user_id = u.id
        LEFT JOIN groups g       ON g.id = ug.group_id
         ORDER BY t.slug, u.email""",
        (BASE_CLEARANCE,),
    )

    collected: dict[str, tuple[str, list[Grant]]] = {}
    for email, tenant_slug, clearance, elevated in cur.fetchall():
        _, grants = collected.setdefault(email, (tenant_slug, []))
        grants.append(Grant(clearance=clearance, elevated_sections=tuple(elevated or ())))

    return {
        email: Principal(email=email, tenant_slug=tenant_slug, grants=tuple(grants))
        for email, (tenant_slug, grants) in collected.items()
    }
