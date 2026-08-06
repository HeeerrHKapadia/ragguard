"""The access policy, expressed as SQL.

access.py is the readable version and stays the authority. This is the same
rules written so the database can apply them during a query instead of
Python applying them afterwards — which is the whole point of Phase 2.
Enforcement has to happen where the rows are selected, because anything
applied later has already let the forbidden content into the process.

Two implementations of one policy is a liability unless they are proven
equal, so scripts/verify_policy_parity.py checks this against the oracle for
every persona-document pair in the corpus. When they disagree, the oracle is
right by definition and this is the thing that gets fixed.
"""

from __future__ import annotations

from ragguard.access import TIER_RANK, Principal


# Inlined rather than defined as a database function so the policy can change
# without a schema migration — and, at this stage, without re-embedding 4996
# chunks to rebuild the volume.
def _tier_case(column: str) -> str:
    return f"""CASE {column}
                 WHEN 'public'       THEN 0
                 WHEN 'internal'     THEN 1
                 WHEN 'confidential' THEN 2
                 WHEN 'restricted'   THEN 3
               END"""


TIER_CASE = _tier_case("d.sensitivity")
CHUNK_TIER_CASE = _tier_case("c.sensitivity")


def _visibility_params(principal: Principal) -> dict:
    sections: list[str] = []
    ranks: list[int] = []
    for grant in principal.grants:
        granted = TIER_RANK[grant.clearance] + 1
        for section in grant.elevated_sections:
            sections.append(section)
            ranks.append(granted)
    return {
        "tenant": principal.tenant_slug,
        "clearance": TIER_RANK[principal.max_clearance],
        "elev_sections": sections,
        "elev_ranks": ranks,
    }


def visibility_sql(principal: Principal) -> tuple[str, dict]:
    """Build a WHERE fragment admitting exactly what this principal may read.

    Returns (sql, params) for psycopg's named-parameter form. The fragment
    assumes `d` is the documents table and `t` is tenants.

    Mirrors can_read() clause for clause:

      1. tenant match, checked first and absolute
      2. clearance rank covers the document's tier, or
      3. a group is elevated in this document's section

    Elevation grants one tier above the group's own baseline, scoped to its
    section — which is what "finance reads confidential finance material"
    means, as opposed to "finance reads confidential anything".
    """
    sql = f"""
        t.slug = %(tenant)s
        AND (
            {TIER_CASE} <= %(clearance)s
            OR EXISTS (
                SELECT 1
                  FROM unnest(
                         %(elev_sections)s::text[],
                         %(elev_ranks)s::int[]
                       ) AS e(sec, rnk)
                 WHERE (
                         d.section = e.sec
                         OR left(d.section, length(e.sec) + 1) = e.sec || '/'
                       )
                   AND e.rnk >= {TIER_CASE}
            )
        )
    """
    return sql, _visibility_params(principal)


def visibility_sql_on_chunks(principal: Principal) -> tuple[str, dict]:
    """Same policy, evaluated on denormalized chunk ACL columns.

    Assumes `c` is chunks and `t` is tenants (for slug). Requires
    ``db/init/03_chunk_acl.sql`` to have populated ``c.section`` and
    ``c.sensitivity``. Prefer this once Phase B is applied so ANN/FTS
    scans do not join documents just to decide visibility.
    """
    sql = f"""
        t.slug = %(tenant)s
        AND (
            {CHUNK_TIER_CASE} <= %(clearance)s
            OR EXISTS (
                SELECT 1
                  FROM unnest(
                         %(elev_sections)s::text[],
                         %(elev_ranks)s::int[]
                       ) AS e(sec, rnk)
                 WHERE (
                         c.section = e.sec
                         OR left(c.section, length(e.sec) + 1) = e.sec || '/'
                       )
                   AND e.rnk >= {CHUNK_TIER_CASE}
            )
        )
    """
    return sql, _visibility_params(principal)
