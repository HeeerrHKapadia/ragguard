"""The access policy, expressed a third time — in Cypher.

There are now three implementations of one policy: access.py as the readable
reference, filters.py for Postgres, and this for Neo4j. Every additional
implementation is another chance for them to diverge, and a divergence here
means the graph returns documents the relational store would have refused.

The mitigation is the same one used for SQL: verify_policy_parity.py checks
all three against every persona-document pair in the corpus, and CI fails on
any disagreement. Three implementations that are proven equal are safer than
two that are merely believed to be.
"""

from __future__ import annotations

from ragguard.access import TIER_RANK, Principal

# Cypher has no ordering over our tier strings, so the ranking is passed in
# as a parameter map and indexed: $tierRank[d.tier]. Embedding the numbers
# in the query text instead would put the policy in two places.
TIER_RANK_PARAM = dict(TIER_RANK)


def visibility_cypher(alias: str = "d") -> str:
    """A Cypher predicate over a Document node, mirroring can_read().

    Tenant is checked first and is absolute, exactly as in the other two
    implementations. The elevation clause lifts a group one tier above its
    baseline within its own section — `STARTS WITH e.sec + '/'` matches a
    nested section without letting `people` grant access to `people-analytics`.
    """
    return f"""
        {alias}.tenant = $tenant
        AND (
            $tierRank[{alias}.tier] <= $clearance
            OR any(e IN $elevated WHERE
                     ({alias}.section = e.sec
                      OR {alias}.section STARTS WITH e.sec + '/')
                 AND e.rnk >= $tierRank[{alias}.tier])
        )
    """


def visibility_params(principal: Principal) -> dict:
    elevated = [
        {"sec": section, "rnk": TIER_RANK[grant.clearance] + 1}
        for grant in principal.grants
        for section in grant.elevated_sections
    ]
    return {
        "tenant": principal.tenant_slug,
        "clearance": TIER_RANK[principal.max_clearance],
        "elevated": elevated,
        "tierRank": TIER_RANK_PARAM,
    }
