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

884 graded cases from 220 distinct queries, scored against 639 documents and
12 personas. The reference retrievers are not systems anyone would ship — they
exist to calibrate the harness, and they bracket the range every real result
must fall inside.

| Retriever | Leak rate | vs ceiling | Local | Global | Worst parity |
| --------- | --------: | ---------: | -----: | -----: | -----------: |
| `null` — returns nothing | 0.0% | 0% | 0.0% | 0.0% | — |
| `leaky` — no filtering at all | 82.5% | 100% | 100.0% | 75.3% | 0.99 |
| `oracle` — perfect and permitted | 0.0% | 100% | 100.0% | 75.5% | 0.99 |
| **Phase 1 — dense, no authorization** | **93.1%** | 83.5% | 92.6% | **25.0%** | 0.88 |

### What Phase 1 found

**The naive retriever leaks more than the retriever built to be reckless.**
93.1% against `leaky`'s 82.5%. `leaky` returns the *relevant* documents
regardless of permission, and relevant documents are often ones the asker is
entitled to. Dense search returns the nearest neighbours across every tenant,
so it reaches for material that is neither relevant nor permitted. Semantic
similarity does not respect organisational boundaries, and nothing about a
document's meaning signals who owns it.

**Dense retrieval is competent at point lookups and hopeless at
cross-document questions:** 92.6% local versus 25.0% global. This is the
premise of the entire GraphRAG argument, measured on this corpus rather than
assumed from a paper. Phase 5 has to beat 25.0% on global queries by enough
to justify several times the token cost — and that target now exists as a
number instead of a hope.

**Approximate search costs 1.27% recall and returns 6.45× the throughput**
(84 → 543 queries/sec, index built in 0.7s). Small enough to take, and now a
measurement rather than a guess. This is why Phase 0a shipped without the
index: exact search had to run first to be the reference.

**Parity is not 1.00 at baseline, and that matters for reading Phase 2.**
No permission filter exists yet, so every persona receives identical results
— yet parity ranges 0.88–1.12. The cause is denominators, not filtering: a
new hire is entitled to 22 documents where an exec is entitled to 300, so the
same absolute miss moves the smaller ratio far more. **Phase 2's parity must
be compared against this 0.88–1.12 band, not against 1.00**, or ordinary
sampling variance will be misread as post-filter collapse.

**Leak rate** — fraction of cases returning any document the persona may not
read. Case-level, not document-level: one forbidden document among ninety-nine
correct ones is a breach, and averaging would report a soothing 1% and bury it.
Target is exactly 0, enforced in CI.

**Recall vs ceiling** — recall as a fraction of what is achievable at this `k`.
A global query with 25 relevant documents caps out at 40% recall when `k=10`,
so raw recall punishes a retriever for a cutoff it did not choose. This is the
number to compare retrievers on.

**Recall parity** — lowest-privilege recall divided by highest-privilege
recall, per tenant, on questions both are entitled to answer. Near 1.0 means
privilege does not degrade quality; below ~0.8 means it does. This is the
metric that catches silent post-filter collapse, and the oracle proves 1.00 is
reachable — so any later drop is a real regression, not a law of nature.

**Over-block rate** — entitled material not returned. Honest caveat: this
conflates filtering with ordinary ranking misses. Phase 2 separates them by
running the identical retriever with and without filtering.

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

Then fetch the corpus and seed it:

```bash
uv run python scripts/fetch_corpus.py
```

```bash
uv run python scripts/seed.py
```

```bash
uv run python scripts/access_matrix.py
```

Then build the golden dataset and calibrate the eval harness:

```bash
uv run python scripts/build_goldens.py
```

```bash
uv run python scripts/evaluate.py
```

## The corpus

Three real, public company handbooks stand in for three tenants — GitLab,
Sourcegraph, and PostHog. Real documents matter here: a synthetic corpus would
let the retrieval look better than it is, because invented documents are
cleanly separated in a way real ones never are.

| Tenant | Documents | Public | Internal | Confidential | Restricted |
| ------ | --------: | -----: | -------: | -----------: | ---------: |
| GitLab | 300 | 22 | 195 | 49 | 34 |
| PostHog | 225 | 28 | 167 | 26 | 4 |
| Sourcegraph | 114 | 7 | 67 | 26 | 14 |
| **Total** | **639** | **57** | **429** | **101** | **52** |

Sensitivity tiers are **derived from the directory each document already lives
in** — `board-meetings` and `acquisitions` become restricted, `finance` and
`legal` become confidential, `engineering` stays internal. These are boundaries
the real companies drew, not ones invented to make a demo work.

The tier distribution is a pyramid because that is what real organisations look
like, and the sampler explicitly refuses to trim the restricted tier: those 52
documents are the entire basis of the leak tests, and public handbooks contain
very few of them (GitLab publishes exactly one board-meeting document, for
obvious reasons).

## Personas

Each tenant has four personas spanning the privilege range. What each can see
is generated by [`scripts/access_matrix.py`](scripts/access_matrix.py):

| Persona | Clearance | Visible (of 300) |
| ------- | --------- | ---------------: |
| `newhire@gitlab.test` | public | 22 |
| `eng@gitlab.test` | internal | 217 |
| `finance@gitlab.test` | internal + finance elevation | 230 |
| `exec@gitlab.test` | restricted | 300 |

The finance persona is the interesting one: it reads **13 of 49** confidential
documents — the ones inside `finance/` and no others. Access in real
organisations is rarely a single global level, and modelling it as one is how
systems end up either over-blocking or over-sharing.

That 22-vs-300 spread between the new hire and the exec is also the core
engineering problem stated in one line. Once retrieval is filtered, the new
hire is searching a corpus one-fourteenth the size — and naive post-filtering
will quietly hand them far worse results while every test still passes.

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
config/
  tenants.yaml       The security model: corpora, tiers, groups, personas
db/init/             SQL run once on first container start
  01_extensions.sql
  02_schema.sql
src/ragguard/
  config.py          Single place that reads the environment
  db.py              Connection helper with pgvector type registration
  corpus.py          Filesystem to documents: frontmatter, tiers, sampling
  access.py          Reference implementation of the policy
  eval/
    dataset.py       Golden cases: local vs global query classes
    metrics.py       Leak rate, recall vs ceiling, recall parity
    retriever.py     Retriever protocol + three calibration retrievers
    runner.py        Corpus + personas + cases -> scored report
scripts/
  smoke_test.py      Phase 0a deliverable
  fetch_corpus.py    Sparse-clone the three handbooks
  corpus_report.py   Dry-run ingestion, check tier distribution
  seed.py            Config to database rows, idempotent
  access_matrix.py   Who can see what — Phase 0b deliverable
  build_goldens.py   Generate eval/goldens.jsonl (committed, diffable)
  evaluate.py        Score retrievers — Phase 0c deliverable
eval/
  goldens.jsonl      884 graded cases. Committed on purpose.
tests/
  test_access.py     The policy is the part that must not silently change
  test_metrics.py    A wrong scorer misleads every result built on it
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

**`access.py` is deliberately the slow, obvious version.** Later phases push
enforcement into SQL filters and an authorization service for speed. The only
way to know those fast paths are correct is to keep a reference implementation
that is obviously correct, and diff against it. When the eval reports a leak,
this is the oracle that says what should have happened.

**Authorization is resolved from the corpus, never from what a retriever
claims.** The scorer originally read each result's tier off the returned
object — meaning a retriever that mislabelled a restricted document as public
would have had its own leak waved through. The component under test was
grading itself. It now resolves every returned URI against the corpus index,
and a URI absent from that index counts as a leak rather than being skipped,
because a citation to a document that does not exist is its own failure.

**The eval harness is calibrated against retrievers with known-correct scores.**
A scorer with an inverted comparison reports encouraging numbers forever and
nobody notices. So before grading anything real, the harness has to prove it
reports 82.5% leakage for a retriever that ignores permissions and a clean
sweep for one that cannot fail. Both checks run in CI.

Writing those calibration retrievers immediately found two bugs I had written:
a "leaky" retriever that was accidentally safe most of the time, and a recall
metric that punished retrievers for the `k` cutoff rather than for missing
documents. Neither would have been visible from a real retriever's numbers.

**The golden dataset is committed and CI verifies it regenerates byte for
byte.** If generation drifts, results from different phases stop being
comparable and the whole table becomes fiction.

**Tenant isolation is checked before anything else, and nothing overrides it.**
It's tempting to model "exec" as simply high-ranking and compare clearance
ranks everywhere — but that collapses two different questions into one. An exec
at GitLab is not slightly-authorized to read PostHog's material; they are
entirely unauthorized. That collapse is how cross-tenant leaks get written.

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
