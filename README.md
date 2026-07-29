# ragguard — a permission-aware GraphRAG knowledge engine

[![ci](https://github.com/HeeerrHKapadia/ragguard/actions/workflows/ci.yml/badge.svg)](https://github.com/HeeerrHKapadia/ragguard/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An enterprise knowledge engine where **who is asking** changes what can be
retrieved — through vectors *and* through a knowledge graph. Multi-tenant,
role-aware, and evaluated for leaks rather than just for relevance.

Most RAG systems answer "what is most relevant?" This one has to answer
"what is most relevant *that this person is allowed to see*?" — including
when the answer is assembled by traversing a graph across many documents —
and prove it, with numbers, under adversarial pressure.

The GraphRAG claim is measured, not assumed: a hybrid-search baseline is
built first and frozen, and the graph must beat it per query class to justify
its 3–10x token cost. Where it loses, the router sends queries to the cheap
path instead.

## Why this problem

Relevance alone leaks. A query from an intern and a query from the CFO produce
the same nearest neighbours; only the authorization layer separates them. The
standard failure modes are all measurable:

- **Post-filtering destroys recall for low-privilege users.** Retrieve top-k,
  then drop what the user can't see → the VP gets 10 results, the intern gets 2.
  Both "work". Only one is useful.
- **Pre-filtering degrades ANN indexes.** Restrictive filters sever HNSW graph
  links; at high selectivity the graph disconnects and recall collapses.
- **Prompt-level enforcement is not enforcement.** "Only use tenant X's
  documents" fails under prompt injection. Filtering must be deterministic,
  in the database, before context reaches the model.
- **Existence leakage.** "I found 3 documents you can't access" leaks that they
  exist. So does a suspiciously specific refusal.
- **Stale ACLs.** Someone changes teams; the index still carries their old
  access. Revocation latency is a measurable security property.

And the graph layer adds four failure modes that vector RAG doesn't have:

- **Edges are themselves sensitive.** Knowing `Ana → works_on → Project Titan`
  leaks that Titan exists and who staffs it, even if every document about it
  is blocked.
- **Community summaries launder sensitivity.** A summary aggregated across
  many documents inherits the sensitivity of *all* of them — serve it to
  everyone and one confidential source has leaked into every answer.
- **Multi-hop traversal escapes the ACL boundary.** Traversal walks from an
  allowed node *through* a forbidden one. Enforcing at every hop can
  disconnect the graph for low-privilege users — recall collapse, but
  structurally worse.
- **Entity resolution can bridge tenants.** If extraction decides "Ana Ruiz"
  in tenant A and tenant B are the same entity, the index itself is a
  cross-tenant leak. Resolution must be per-tenant by construction.

## Results

<!-- Filled in from Phase 1 onward. This table is the point of the project. -->

| Stage | Leak rate | Over-block rate | Recall parity | Faithfulness | p95 latency |
| ----- | --------- | --------------- | ------------- | ------------ | ----------- |
| Phase 1 — naive baseline | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

**Leak rate** — authorized-to-see violations. Target: exactly 0, enforced in CI.
**Over-block rate** — content wrongly withheld. A system that answers nothing
never leaks and is worthless.
**Recall parity** — retrieval quality for a low-privilege persona divided by
that of a high-privilege persona, on questions *both* are entitled to answer.
Target ≈ 1.0. This is the metric that catches silent post-filter collapse.

## Stack

| Layer | Choice | Why |
| ----- | ------ | --- |
| Vector + relational | Postgres 17 + pgvector | One database for vectors, lexical search, and ACL metadata |
| Lexical | Postgres `tsvector` | BM25-ish half of hybrid retrieval |
| Graph | Neo4j (Phase 4+) | Knowledge graph with per-node/per-edge ACL metadata |
| Runtime | Python 3.13, `uv` | |
| Later phases | Qdrant, OpenFGA, Langfuse, FastAPI | Added only when a failing metric demands them |

## Setup

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
```

```bash
docker compose up -d
```

```bash
uv sync
```

```bash
uv run python scripts/smoke_test.py
```

The smoke test verifies connectivity, the pgvector extension, the schema, write
paths, similarity **ranking** (against known vector geometry, not just "rows came
back"), full-text search, and tenant isolation. It runs in a transaction that is
rolled back, so it is repeatable and leaves no residue.

### Resetting the database

Scripts in `db/init/` run **once**, on first startup of an empty volume. After
editing them:

```bash
docker compose down -v
```

```bash
docker compose up -d
```

## Layout

```
db/init/          SQL run once on first container start
  01_extensions.sql
  02_schema.sql
src/ragguard/     Application package
  config.py       Single place that reads the environment
  db.py           Connection helper with pgvector type registration
scripts/
  smoke_test.py   Phase 0a deliverable
```

## Design notes

**`chunks.tenant_id` is denormalized on purpose.** It's derivable from
`document_id` via a join, but every retrieval query must filter by tenant, and a
join inside a filtered vector search is exactly what wrecks recall and latency.
The filter column has to live on the row being scanned.

**The HNSW index is deliberately absent in Phase 0a.** Without it, Postgres does
exact nearest-neighbour search — perfect recall, fine at small scale. It gets
added in Phase 1 so the cost of approximation is *measured*, and again in
Phase 3 to measure what permission filters cost on top. Creating it now would
hide both lessons behind "it just worked".

**The eval harness is built before the RAG pipeline.** Phase 1 is a baseline
designed to fail, so there's an honest before-picture to improve on.

## Roadmap

- **0a** Infra foundations — Docker Compose, Postgres, pgvector *(in progress)*
- **0b** Corpus + identity model — real public company handbooks as tenants
- **0c** Eval harness, built *before* any retrieval exists — with local vs
  global query classes, since GraphRAG only earns its cost on the latter
- **1** Deliberately naive baseline that leaks
- **2** Move enforcement into the retrieval layer; drive leak rate to zero
- **3** Hybrid + reranking (pgvector vs Qdrant) — **the control group** the
  graph must beat
- **4** Knowledge graph construction — LLM extraction → Neo4j, ACL metadata
  on every node and edge, per-tenant entity resolution
- **5** Graph retrieval + local/global query routing, head-to-head vs Phase 3
- **6** Permission-aware graph traversal — edge sensitivity, summary
  laundering, hop-boundary enforcement *(the novel part)*
- **7** Real authorization layer (OpenFGA / ReBAC) across both stores
- **8** Adversarial hardening + red-team regression suite
- **9** Ship — API, tracing, CI gates, persona + graph-visualization demo
