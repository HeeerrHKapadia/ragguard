"""Prove every fast path agrees with the reference implementation.

Phase 0b kept access.py deliberately slow and obvious so that fast paths
could be diffed against something known-correct. This is that diff, and
there are now two of them: SQL for Postgres and Cypher for Neo4j.

Every persona is checked against every document — 12 x 639 comparisons per
implementation — and both predicates must return exactly the set the oracle
permits. Not approximately, not for a sample: exactly, for all of them.

Three implementations of one policy is three chances to diverge, and a
divergence between the stores is the worst kind: the graph would return
documents the relational store refuses, so which answer a user gets would
depend on which retrieval path happened to run.

Two disagreements are possible and they are not equally bad. Admitting
something the oracle forbids is a leak. Rejecting something the oracle
allows is over-blocking. Both are reported, because a policy that silently
hides material a user is entitled to is broken in a way no leak metric will
ever catch.

Run:  uv run python scripts/verify_policy_parity.py
"""

from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import can_read, load_principals
from ragguard.db import connect
from ragguard.graph.filters import visibility_cypher, visibility_params
from ragguard.graph.store import graph_driver
from ragguard.retrieval.filters import visibility_sql


def main() -> int:
    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)

            cur.execute(
                """SELECT d.source_uri, t.slug, d.section, d.sensitivity
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
            )
            documents = cur.fetchall()

            if not documents or not principals:
                print("Seed the database first: uv run python scripts/seed.py")
                return 1

            comparisons = len(principals) * len(documents)
            print(f"\n{len(principals)} personas x {len(documents)} documents "
                  f"= {comparisons} comparisons per implementation\n")

            graph_available = True
            try:
                driver_ctx = graph_driver()
                driver = driver_ctx.__enter__()
                driver.verify_connectivity()
                session = driver.session()
            except Exception as exc:  # noqa: BLE001 - any driver failure is the same story
                print(f"  Neo4j unavailable, checking SQL only: {str(exc).strip()[:80]}")
                graph_available = False

            # An empty graph makes every Cypher query return nothing, which
            # this script would otherwise report as the policy blocking all
            # 1729 permitted documents. That reads like a catastrophic policy
            # bug and is actually a missing build step — worth distinguishing,
            # because the two have nothing to do with each other.
            if graph_available:
                node_count = session.run(
                    "MATCH (d:Document) RETURN count(d) AS n"
                ).single()["n"]
                if node_count == 0:
                    print("\n  Neo4j is reachable but holds no documents.")
                    print("  Build the graph first: uv run python scripts/build_graph.py\n")
                    session.close()
                    driver_ctx.__exit__(None, None, None)
                    return 1

            problems: dict[str, int] = {"sql-leak": 0, "sql-block": 0,
                                        "cypher-leak": 0, "cypher-block": 0}

            for email in sorted(principals):
                principal = principals[email]

                oracle = {
                    uri for uri, tenant, section, tier in documents
                    if can_read(principal, tenant, section, tier)
                }

                where, params = visibility_sql(principal)
                cur.execute(
                    f"""SELECT d.source_uri
                          FROM documents d JOIN tenants t ON t.id = d.tenant_id
                         WHERE {where}""",
                    params,
                )
                from_sql = {row[0] for row in cur.fetchall()}

                from_cypher: set[str] | None = None
                if graph_available:
                    result = session.run(
                        f"MATCH (d:Document) WHERE {visibility_cypher('d')} RETURN d.uri AS uri",
                        **visibility_params(principal),
                    )
                    from_cypher = {record["uri"] for record in result}

                marks = []
                for label, produced in (("sql", from_sql), ("cypher", from_cypher)):
                    if produced is None:
                        marks.append("--")
                        continue
                    leaks = produced - oracle
                    blocks = oracle - produced
                    problems[f"{label}-leak"] += len(leaks)
                    problems[f"{label}-block"] += len(blocks)
                    marks.append("ok" if not (leaks or blocks) else "MISMATCH")

                    for uri in sorted(leaks)[:2]:
                        print(f"      {label} admits, oracle forbids : {uri}")
                    for uri in sorted(blocks)[:2]:
                        print(f"      {label} blocks, oracle allows  : {uri}")

                print(f"  {email:<28} {len(oracle):>4} visible   "
                      f"sql={marks[0]:<9} cypher={marks[1]}")

            if graph_available:
                session.close()
                driver_ctx.__exit__(None, None, None)

    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    print()
    if any(problems.values()):
        for key, count in problems.items():
            if count:
                print(f"POLICY MISMATCH: {key} = {count}")
        print("The oracle is correct by definition. Fix the fast path.")
        return 1

    checked = "SQL and Cypher policies are" if graph_available else "SQL policy is"
    print(f"{checked} identical to the reference implementation.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
