"""Leaks that document-level permission checks cannot see.

Phase 5 measured a leak rate of 0.0% for graph retrieval. Every document
returned was one the user was entitled to read, and by the standard of every
metric built so far the system is clean.

It is not. Filtering the endpoint of a traversal says nothing about what the
traversal walked through to get there, and a path is information even when
its destination is permitted.

Three failures, none of which appear in a document-level leak rate:

**Transit.** A path from a permitted seed to a permitted result can pass
through a document the user may not read. The forbidden document never
appears in the output, but it selected the output — it is the reason those
particular results surfaced. Expose the path as an explanation, which any
graph system eventually does, and the leak becomes literal.

**Concept disclosure.** A concept is a node whose name is a summary of the
documents that mention it. When every one of those documents is forbidden,
the concept's existence discloses that restricted material exists and what
it concerns. Nothing about that requires reading the document.

**Inferred existence.** Degree is information. Learning that a permitted
document has six neighbours while only two are visible tells the user four
documents exist that they cannot see, and roughly what they relate to.

The measurement comes first. A fix whose cost is not known against a leak
whose size is not known is a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragguard.graph.filters import visibility_cypher

# A concept is legitimately visible when the principal can read at least one
# document that mentions it. Checking against a stored max_tier would be
# cheaper and wrong: a concept named by both an internal and a restricted
# document is fine to surface, because the internal document already
# entitles the user to know the term exists.
CONCEPT_VISIBLE = f"""
    exists {{
        MATCH (concept)<-[:MENTIONS]-(witness:Document)
        WHERE {visibility_cypher('witness')}
    }}
"""

# Walks every path the unguarded retriever would walk, then inspects the
# intermediate nodes it never checked. Returns one row per path so distinct
# counting happens in Python — Neo4j Community has no APOC, and hand-rolling
# set aggregation in Cypher would obscure what is being counted.
AUDIT_CYPHER = f"""
UNWIND $seeds AS seed
MATCH p = (s:Document {{uri: seed.uri}})-[:LINKS_TO|MENTIONS*1..2]-(n:Document)
WHERE n.uri <> s.uri AND {visibility_cypher('n')}
WITH p, n,
     [node IN nodes(p)
      WHERE node:Document AND NOT ({visibility_cypher('node')}) | node.uri] AS blocked_docs,
     [concept IN nodes(p)
      WHERE concept:Concept AND NOT ({CONCEPT_VISIBLE}) | concept.name] AS blocked_concepts
RETURN n.uri AS result_uri, blocked_docs, blocked_concepts
"""

# What a low-privilege user can infer about what they cannot see, purely
# from how many neighbours a permitted document turns out to have.
DEGREE_CYPHER = f"""
UNWIND $seeds AS seed
MATCH (s:Document {{uri: seed.uri}})
MATCH (s)-[:LINKS_TO]-(n:Document)
WITH s,
     count(n) AS total,
     sum(CASE WHEN {visibility_cypher('n')} THEN 1 ELSE 0 END) AS visible
WHERE total > visible
RETURN sum(total - visible) AS hidden_neighbours, count(s) AS documents
"""


@dataclass
class AuditResult:
    paths: int = 0
    transit_paths: int = 0
    concept_paths: int = 0
    blocked_documents: set[str] = field(default_factory=set)
    blocked_concepts: set[str] = field(default_factory=set)
    hidden_neighbours: int = 0

    @property
    def transit_rate(self) -> float:
        return self.transit_paths / self.paths if self.paths else 0.0

    @property
    def concept_rate(self) -> float:
        return self.concept_paths / self.paths if self.paths else 0.0
