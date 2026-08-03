"""Graph-augmented retrieval.

Dense search answers "which documents look like this query?" That works when
one document holds the answer and fails when the answer is distributed —
which is exactly what the global query class is. Phase 1 measured the gap:
92.6% local against 25.0% global.

The graph attacks the second case by using the first as a starting point.
Dense search picks entry documents, then traversal collects what those
documents are connected to. A page nobody would rank highly for the query
itself becomes reachable because a page that *did* rank links to it, or names
the same concept.

Two edge types carry the expansion:

  LINKS_TO   an author explicitly connected these pages
  MENTIONS   two pages name the same concept

The first is precise and sparse; the second is looser and dense. Together
they reach documents dense search cannot, which is the whole hypothesis
Phase 5 exists to test — including the possibility that it fails.
"""

from __future__ import annotations

from neo4j import Driver

from ragguard.access import Principal
from ragguard.eval.dataset import GoldenCase
from ragguard.eval.metrics import RetrievedDoc
from ragguard.graph.audit import CONCEPT_VISIBLE
from ragguard.graph.filters import visibility_cypher, visibility_params
from ragguard.retrieval.fusion import reciprocal_rank_fusion

# How many dense hits seed the traversal. Too few and expansion inherits
# their blind spots; too many and everything is within reach of something,
# so the graph stops discriminating.
SEED_K = 10

# Ceiling on expanded candidates. A concept can name up to 40 documents, so
# ten seeds can reach several hundred nodes without a bound.
EXPANSION_LIMIT = 60

EXPAND_CYPHER = f"""
UNWIND $seeds AS seed
MATCH (s:Document {{uri: seed.uri}})
MATCH (s)-[:LINKS_TO|MENTIONS*1..2]-(n:Document)
WHERE n.uri <> s.uri AND {visibility_cypher('n')}
RETURN n.uri              AS uri,
       count(DISTINCT s.uri) AS seed_hits,
       min(seed.rank)     AS best_rank
ORDER BY seed_hits DESC, best_rank ASC
LIMIT $limit
"""

# The same traversal, with every node on the path checked rather than only
# the destination.
#
# `all(node IN nodes(p) ...)` is the entire difference. Without it a path can
# run from a permitted seed, through a document the user may not read, to a
# permitted result — and the forbidden document, invisible in the output, is
# what selected that result. Measured at 5.8% of paths overall and 64.3% for
# the least privileged persona, against a document-level leak rate of 0.0%.
#
# Concepts are checked by witness rather than by stored tier: a concept is
# visible when the user can read at least one document that mentions it.
# A concept named by both internal and restricted documents is legitimate to
# traverse, because the internal document already entitles the user to know
# the term exists.
GUARDED_EXPAND_CYPHER = f"""
UNWIND $seeds AS seed
MATCH p = (s:Document {{uri: seed.uri}})-[:LINKS_TO|MENTIONS*1..2]-(n:Document)
WHERE n.uri <> s.uri
  AND all(node IN nodes(p) WHERE
        CASE
          WHEN node:Concept  THEN {CONCEPT_VISIBLE}
          WHEN node:Document THEN {visibility_cypher('node')}
          ELSE false
        END)
RETURN n.uri              AS uri,
       count(DISTINCT s.uri) AS seed_hits,
       min(seed.rank)     AS best_rank
ORDER BY seed_hits DESC, best_rank ASC
LIMIT $limit
"""


class GraphRetriever:
    """Dense seeds, expanded through the graph, fused.

    The permission filter is applied inside the traversal rather than to its
    output. Phase 2 established that filtering during retrieval beats
    filtering after it, and the argument is stronger here: an expanded set
    trimmed afterwards would have spent its budget reaching documents that
    were always going to be discarded.
    """

    name = "graph"

    def __init__(self, seeder, driver: Driver, seed_k: int = SEED_K,
                 weights: tuple[float, float] = (1.0, 1.0),
                 guarded: bool = False) -> None:
        self.seeder = seeder
        self.driver = driver
        self.seed_k = seed_k
        self.weights = weights
        self.guarded = guarded
        self.cypher = GUARDED_EXPAND_CYPHER if guarded else EXPAND_CYPHER
        self.name = "graph-guarded" if guarded else "graph"

    def retrieve(self, case: GoldenCase, principal: Principal, k: int) -> list[RetrievedDoc]:
        seeds = self.seeder.retrieve(case, principal, self.seed_k)
        if not seeds:
            return []

        seed_rows = [{"uri": doc.uri, "rank": i} for i, doc in enumerate(seeds, start=1)]
        params = visibility_params(principal)
        params["seeds"] = seed_rows
        params["limit"] = EXPANSION_LIMIT

        with self.driver.session() as session:
            expanded = [
                record["uri"]
                for record in session.run(self.cypher, **params)
            ]

        seed_ranking = [doc.uri for doc in seeds]
        fused = reciprocal_rank_fusion(
            [seed_ranking, expanded], weights=list(self.weights)
        )[:k]

        # Seeds already carry correct metadata; expanded documents need it
        # looked up. The scorer re-resolves everything against the corpus
        # regardless, so this is for callers outside the harness.
        known = {doc.uri: doc for doc in seeds}
        missing = [uri for uri, _ in fused if uri not in known]
        if missing:
            with self.driver.session() as session:
                for record in session.run(
                    "MATCH (d:Document) WHERE d.uri IN $uris "
                    "RETURN d.uri AS uri, d.tenant AS tenant, "
                    "       d.section AS section, d.tier AS tier",
                    uris=missing,
                ):
                    known[record["uri"]] = RetrievedDoc(
                        uri=record["uri"], tenant=record["tenant"],
                        section=record["section"], tier=record["tier"],
                    )

        out: list[RetrievedDoc] = []
        for uri, score in fused:
            doc = known.get(uri)
            if doc is not None:
                out.append(
                    RetrievedDoc(uri=doc.uri, tenant=doc.tenant, section=doc.section,
                                 tier=doc.tier, score=score)
                )
        return out
