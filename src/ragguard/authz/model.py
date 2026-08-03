"""The access policy as a relationship graph.

Zanzibar-style systems answer one question: is there a path of relationships
from this user to this object? They do not compare values. Our policy does —
"clearance rank at least the document's tier" is an inequality, and there is
no relationship for `>=`.

The resolution is to materialise the ordering as relationships. Tiers form a
chain, each pointing at the next stricter one, and `cleared` inherits along
it. A group cleared at `restricted` is therefore cleared at `confidential`
by inheritance rather than by comparison, and the ladder that was arithmetic
in access.py becomes four nodes and three edges here.

Section elevation gets the same treatment. "Finance may read confidential
material inside finance" is not a tier comparison scoped by a string prefix;
it is a grant object named `finance/confidential` that the finance group
holds and that finance's confidential documents point to. A document is
readable when the user holds the exact grant it names.

**Every object id is namespaced by tenant** — `document:gitlab::values.md`,
`group:gitlab::engineering`. Cross-tenant access is not forbidden by a rule
that could be forgotten; there is simply no path, because a GitLab group and
a PostHog document share no object in common. Same tactic as concept keys in
the graph, for the same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from ragguard.access import TIER_RANK

TIERS = ["public", "internal", "confidential", "restricted"]

# The model, in the DSL that OpenFGA's docs and playground use. Kept as the
# readable form even though it is not what gets uploaded: the Python SDK has
# no DSL transformer (that lives in the JS SDK and the CLI), so build_model()
# below constructs the equivalent typed objects. This string is the spec that
# function is checked against by eye.
#
# `cleared` accepts a bare user as well as a group's members so that a person
# with no group membership still resolves to public access, matching
# BASE_CLEARANCE in the reference implementation.
AUTH_MODEL_DSL = """
model
  schema 1.1

type user

type group
  relations
    define member: [user]

type tier
  relations
    define stricter: [tier]
    define cleared: [user, group#member] or cleared from stricter

type grant
  relations
    define holder: [group#member]

type document
  relations
    define tier: [tier]
    define grant: [grant]
    define viewer: cleared from tier or holder from grant
"""


def build_model():
    """The DSL above, as SDK objects.

    `tupleToUserset` is the piece doing the real work. "cleared from tier"
    means: follow this document's `tier` relation to a tier object, then ask
    whether the user is `cleared` on that object. Chained through the tier
    ladder, it turns an inequality into graph traversal.
    """
    from openfga_sdk.models import (
        Metadata,
        ObjectRelation,
        RelationMetadata,
        RelationReference,
        TupleToUserset,
        TypeDefinition,
        Userset,
        Usersets,
    )

    direct = Userset(this={})

    def from_relation(tupleset: str, computed: str) -> Userset:
        return Userset(tuple_to_userset=TupleToUserset(
            tupleset=ObjectRelation(relation=tupleset),
            computed_userset=ObjectRelation(relation=computed),
        ))

    user_ref = RelationReference(type="user")
    group_members = RelationReference(type="group", relation="member")
    tier_ref = RelationReference(type="tier")
    grant_ref = RelationReference(type="grant")

    return [
        TypeDefinition(type="user", relations={}, metadata=Metadata(relations={})),

        TypeDefinition(
            type="group",
            relations={"member": direct},
            metadata=Metadata(relations={
                "member": RelationMetadata(directly_related_user_types=[user_ref]),
            }),
        ),

        TypeDefinition(
            type="tier",
            relations={
                "stricter": direct,
                "cleared": Userset(union=Usersets(child=[
                    direct,
                    from_relation("stricter", "cleared"),
                ])),
            },
            metadata=Metadata(relations={
                "stricter": RelationMetadata(directly_related_user_types=[tier_ref]),
                "cleared": RelationMetadata(
                    directly_related_user_types=[user_ref, group_members]
                ),
            }),
        ),

        TypeDefinition(
            type="grant",
            relations={"holder": direct},
            metadata=Metadata(relations={
                "holder": RelationMetadata(directly_related_user_types=[group_members]),
            }),
        ),

        TypeDefinition(
            type="document",
            relations={
                "tier": direct,
                "grant": direct,
                "viewer": Userset(union=Usersets(child=[
                    from_relation("tier", "cleared"),
                    from_relation("grant", "holder"),
                ])),
            },
            metadata=Metadata(relations={
                "tier": RelationMetadata(directly_related_user_types=[tier_ref]),
                "grant": RelationMetadata(directly_related_user_types=[grant_ref]),
            }),
        ),
    ]


# OpenFGA splits `type:id` on the colon, so an id containing one is
# malformed — which rules out both a `::` namespace separator and the raw
# document URIs, since those are `gitlab://values.md`. Anything outside this
# set is replaced rather than escaped: ids only need to be unique and
# readable in error messages, not reversible.
UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_.@/+-]")

TENANT_SEP = "--"


def safe_id(raw: str) -> str:
    return UNSAFE_ID_CHARS.sub("_", raw)


def scoped(kind: str, tenant: str, name: str) -> str:
    """Tenant-namespaced object id. The isolation boundary, structurally.

    A GitLab group and a PostHog document share no object, so cross-tenant
    access has no path to traverse. Not a rule that could be forgotten — an
    absence of edges.
    """
    return f"{kind}:{safe_id(tenant)}{TENANT_SEP}{safe_id(name)}"


@dataclass(frozen=True)
class Tuple:
    user: str
    relation: str
    object: str

    def as_dict(self) -> dict:
        return {"user": self.user, "relation": self.relation, "object": self.object}


def tier_chain(tenant: str) -> list[Tuple]:
    """Link each tier to the next stricter one.

    `cleared` inherits from `stricter`, so clearance at a high tier flows
    down to every tier below it. Three tuples replace an integer comparison.
    """
    out = []
    for looser, stricter in pairwise(TIERS):
        out.append(Tuple(
            user=scoped("tier", tenant, stricter),
            relation="stricter",
            object=scoped("tier", tenant, looser),
        ))
    return out


def grant_name(section: str, tier: str) -> str:
    return f"{section}/{tier}"


def identity_tuples(tenant: str, groups: list[dict], users: list[dict]) -> list[Tuple]:
    """Memberships, clearances, and section grants."""
    out: list[Tuple] = []

    for group in groups:
        group_obj = scoped("group", tenant, group["slug"])
        members = f"{group_obj}#member"

        out.append(Tuple(
            user=members,
            relation="cleared",
            object=scoped("tier", tenant, group.get("clearance", "internal")),
        ))

        # Elevation lifts the group one tier above its own baseline, inside
        # its own sections — expressed as holding that exact grant.
        elevated_tier = TIERS[min(
            TIER_RANK[group.get("clearance", "internal")] + 1, len(TIERS) - 1
        )]
        for section in group.get("elevated", []):
            out.append(Tuple(
                user=members,
                relation="holder",
                object=scoped("grant", tenant, grant_name(section, elevated_tier)),
            ))

    for entry in users:
        user_obj = scoped("user", tenant, entry["email"])
        # Baseline: every authenticated tenant member reads public material,
        # with or without a group.
        out.append(Tuple(
            user=user_obj,
            relation="cleared",
            object=scoped("tier", tenant, "public"),
        ))
        for slug in entry.get("groups", []):
            out.append(Tuple(
                user=user_obj,
                relation="member",
                object=scoped("group", tenant, slug),
            ))

    return out


def document_tuples(tenant: str, documents: list[tuple[str, str, str]]) -> list[Tuple]:
    """Point each document at the tier and grant that would permit it.

    `documents` is (source_uri, section, tier).
    """
    out: list[Tuple] = []
    for uri, section, tier in documents:
        doc_obj = scoped("document", tenant, uri)
        out.append(Tuple(
            user=scoped("tier", tenant, tier),
            relation="tier",
            object=doc_obj,
        ))
        out.append(Tuple(
            user=scoped("grant", tenant, grant_name(section, tier)),
            relation="grant",
            object=doc_obj,
        ))
    return out
