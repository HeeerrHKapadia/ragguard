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

## Run the whole thing with one command

```bash
docker run -p 7860:7860 ghcr.io/heeerrhkapadia/ragguard:latest
```

Then open `http://localhost:7860`.

No database to provision, no corpus to download, no API key, no
configuration file. The image carries Postgres with pgvector, all 4,996
embedded chunks, the 384-dimension ONNX embedding model, the API and the
demo UI. Nothing reaches the network at request time.

That is a deliberate inversion. Seeding and embedding happen at **build**
time and are baked into the image layers, so the build takes about fifteen
minutes and the container starts in seconds with data already present. The
usual arrangement — small image, provision a database, run migrations, load
data — is better for a system that changes. This one is a fixed demonstration
over a pinned corpus, so paying once at build beats paying on every start.

It also makes the whole thing free to host, which was the actual constraint:

| | |
| --- | --- |
| Image hosting | GitHub Container Registry, free for public images |
| Build minutes | GitHub Actions, unmetered on public repositories |
| Running it | Hugging Face Spaces free tier, or any machine with Docker |
| Credentials needed | none |

The published image is verified before it is tagged: the workflow starts it,
waits for `/api/health`, and asserts the corpus is actually there. An image
that boots but holds no documents is worse than a failed build, because it
looks fine until someone searches.

## Documentation

| | |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data flow, and the decisions behind them |
| [WALKTHROUGH.md](docs/WALKTHROUGH.md) | What was measured, and the four predictions that were wrong |
| [DEPLOY.md](docs/DEPLOY.md) | Deploying to Fly.io, and what to check when it misbehaves |
| [IMPROVEMENT_PLAN.md](docs/IMPROVEMENT_PLAN.md) | Ranked roadmap to make the system faster and scale cleanly |

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

Phase 2 measured all three under identical conditions, with the HNSW index in
place (Phase 1's published figures used exact search, which is why the naive
row shifts):

| Retriever | Leak rate | vs ceiling | Local | Global | Throughput |
| --------- | --------: | ---------: | -----: | -----: | ---------: |
| dense, no authorization | 92.9% | 82.2% | 91.1% | 24.8% | 371 q/s |
| **post-filter** — retrieve, then drop | **0.0%** | 84.5% | 92.6% | 30.4% | 409 q/s |
| **pre-filter** — never consider | **0.0%** | **86.8%** | **94.6%** | **32.9%** | 199 q/s |

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

### Filtered search: pgvector vs Qdrant

The benchmark deferred from Phase 3, and the one that corrected an earlier
claim of mine.

| Persona | Visible | pgvector recall | pgvector | Qdrant recall | Qdrant |
| ------- | ------: | --------------: | -------: | ------------: | -----: |
| exec@gitlab | 300 | 100.0% | 5.5 ms | 100.0% | 15.8 ms |
| finance@gitlab | 230 | 100.0% | 4.2 ms | 100.0% | 19.7 ms |
| eng@gitlab | 217 | 100.0% | 4.1 ms | 100.0% | 20.2 ms |
| newhire@gitlab | 22 | 100.0% | **1.0 ms** | 100.0% | 25.4 ms |

Recall is measured against exact sequential scan over the same permitted
set, so 100% means nothing was missed.

#### What `EXPLAIN` showed, and the Phase 2 claim it corrects

Phase 2 reported that pre-filtering halved throughput and explained it as a
selective `WHERE` degrading Postgres's use of the HNSW index. The query plans
say something sharper:

```
unfiltered   Index Scan using chunks_embedding_hnsw          12.0 ms
filtered     Bitmap Heap Scan → Index Scan → top-N heapsort   7.3 ms  (300 visible)
filtered     Bitmap Heap Scan → Index Scan → top-N heapsort   0.5 ms  ( 22 visible)
```

Postgres does not degrade the HNSW index under a filter. It **abandons it**
and performs exact search over the permitted rows. That explains all three
observations at once: recall is 100% because the search is exact, a more
selective filter is *faster* rather than slower because fewer rows are
sorted, and Phase 2's "halving" was the planner switching strategies rather
than an index performing badly.

So at this corpus size the HNSW index contributes nothing to the production
query path, because every real query carries a permission filter and none of
them use it.

#### The first version of this benchmark was unfair

Qdrant looked 4–25× slower, and the comparison was rigged without meaning to
be: pgvector answered on an already-open local socket while Qdrant paid HTTP
framing on every request. At single-digit milliseconds that overhead is the
same magnitude as the difference being measured.

Switching Qdrant to gRPC moved narrow-filter latency from **9.2ms to 1.1ms**
at 5,000 vectors. Most of the original gap was transport, not index.

#### The crossover, with transport equalised

Scaled by perturbing real embeddings and renormalising rather than sampling
random ones — HNSW performance depends on clustering, and uniform vectors
are a pathological case that would flatter exact scan.

| Vectors | pgvector broad | Qdrant broad | pgvector narrow | Qdrant narrow |
| ------: | -------------: | -----------: | --------------: | ------------: |
| 25,000 | 1.3 ms | 1.8 ms | 1.3 ms | **1.2 ms** |
| 100,000 | 2.1 ms | 2.2 ms | 1.3 ms | 1.3 ms |
| 250,000 | 3.1 ms | **1.9 ms** | 2.8 ms | **2.0 ms** |
| 500,000 | 6.5 ms | **5.3 ms** | 4.7 ms | 4.7 ms |

pgvector grows roughly with the permitted-set size, which is what exact scan
predicts. Qdrant grows more slowly. The lines cross somewhere between 25,000
and 250,000 vectors depending on how selective the filter is.

**Two caveats, because the numbers are closer than the table suggests.** The
narrow-filter crossover at 25,000 is 1.2ms against 1.3ms — inside the noise,
and not a result worth quoting on its own. The broad-filter crossing at
250,000 (1.9ms against 3.1ms) is the defensible one. Qdrant's jump from
1.9ms to 5.3ms between 250,000 and 500,000 is larger than the trend explains,
which says 20 probe queries is a thin sample.

Absolute differences stay under 7ms throughout. Neither engine is slow at
these sizes; the question is only which grows faster.

**Qdrant is still not used in the retrieval path**, and that decision now has
a stated boundary rather than resting on a mismeasurement. At this project's
actual scale — 4,996 vectors — pgvector wins and the extra service earns
nothing. Above roughly a quarter of a million vectors with permission filters
on every query, the conclusion inverts and this benchmark is the thing to
re-run. The Qdrant filter was verified identical to the oracle across all
twelve personas first, so the choice is about latency and not correctness.

### Phase 9 — the service

```bash
uv run uvicorn ragguard.api.app:app --port 8000
```

Then open `http://localhost:8000` and switch personas.

**The demo is one query asked by three people.** *"compensation review and
pay bands"*, same index, same code path:

| Persona | Top results |
| ------- | ----------- |
| `newhire@gitlab` | GitLab Communication · Performance Indicator Working Group · Measuring Impact — all `public`, nothing about compensation |
| `eng@gitlab` | Compensation at GitLab · Guide to Total Rewards · Incentives at GitLab — all `internal` |
| `exec@gitlab` | **Compensation Review Conversations** (`restricted`) · Talent Assessment (`confidential`) · Corporate FP&A (`confidential`) |

Zero of eight results are shared between the new hire and the executive.

**Identity comes from the bearer token and nothing else.** No endpoint
accepts a persona, tenant, or clearance from a request body — if one did,
changing what you can read would be a matter of editing a JSON field.

Token handling is tested against forgery directly: wrong signature, the
`alg: none` attack, expired tokens, and tokens with no subject. Every
rejection returns an identical message, because distinguishing "expired"
from "forged" from "absent" is a small courtesy to a legitimate user and a
useful oracle to everyone else.

**The `/api/graph` endpoint deliberately does not report how many neighbours
were withheld.** That count is the existence-disclosure risk recorded in
Phase 8, and an endpoint serving it would promote a latent weakness into a
supported feature.

Tracing is local rather than Langfuse — spans, a ring buffer, and an
endpoint — because Langfuse needs an account and the whole project runs
without credentials. The traces record where time went and deliberately do
not record what was hidden.

### Phase 8 — red team

Eight attacks, run in CI. A breach fails the build; a known-open risk is
recorded rather than quietly closed.

| | Attack | Result |
| --- | --- | --- |
| A1 | Cross-tenant extraction by query crafting | 120 crafted queries, 0 foreign documents |
| A2 | Tier escalation using known titles | 101 exact-title searches, 0 hits |
| A3 | Existence disclosure via result counts | **open risk** |
| A4 | Stale permissions after revocation | 217 → 22 on revoke, restored cleanly |
| A5 | Graph transit through forbidden documents | guard holds; 529 paths would leak unguarded |
| A6 | Concept nodes bridging tenants | none |
| A7 | Identifier collision in the authz store | **found and fixed** |
| A8 | One document dominating every result | top document in 15% of result sets |

**A2 is the most realistic attack in the set** — an employee hears a document
mentioned in a meeting and searches for it by name. 101 exact-title searches
against documents each persona is specifically forbidden from reading, zero
hits.

#### A7: the suite found a vulnerability in Phase 7's own code

OpenFGA ids cannot contain a colon, so document URIs are rewritten. That
rewrite was **not injective**: `a:b.md` and `a_b.md` both became `a_b.md`,
which means two documents sharing one authorization object — and therefore
each other's permissions, the more privileged silently granting access to
the less.

It happened not to fire on this corpus. All 639 ids were distinct, so every
test passed and the collision check on real data still passes today. That is
luck, not design, and it would have become a real breach the first time a
document URI contained a colon in a different position.

Fixed by appending a digest of the original whenever a rewrite occurs, so
ids that need no rewriting stay readable and the rest stay unique. The
adversarial-input check is now part of the suite.

#### A3: an open risk, deliberately

Result counts differ by privilege — 28 for the lowest-privilege persona
against 40 for an executive on the same queries. The count alone signals how
much is being withheld.

Not fixed, because the obvious fix does not work. Padding responses to a
fixed size requires having something to pad with, and a persona entitled to
seven documents cannot be padded to ten without either repeating results or
returning documents they may not see. Recording it as a known weakness is
more honest than a mitigation that only obscures it.

### Phase 7 — a real authorization service

Four implementations of one policy now: the Python oracle, a SQL predicate,
a Cypher predicate, and a relationship graph in OpenFGA. All four verified
identical against every persona-document pair, in CI.

#### Modelling an ordered policy in a system with no ordering

Zanzibar-style authorization answers one question — is there a path of
relationships from this user to this object? It does not compare values, and
"clearance rank at least the document's tier" is an inequality.

The fix is to make the ordering a relationship. Tiers form a chain, each
pointing at the next stricter one, and `cleared` inherits along it:

```
type tier
  relations
    define stricter: [tier]
    define cleared: [user, group#member] or cleared from stricter
```

A group cleared at `restricted` is cleared at `confidential` by inheritance
rather than arithmetic. The integer ladder in `access.py` becomes four nodes
and three edges.

Section elevation gets the same treatment: "finance may read confidential
material inside finance" is not a scoped comparison but a grant object named
`finance/confidential` that the group holds and that finance's confidential
documents point at.

**Tenant isolation is again structural.** Every object id is namespaced —
`document:gitlab--values.md`, `group:gitlab--engineering` — so a GitLab group
and a PostHog document share no object and there is no path to traverse.

#### The architectural question, measured

The standard advice for putting an authorization service in front of search
is to retrieve candidates, then check each result. That is post-filtering,
and Phase 2 measured it costing the least privileged persona 14.3 points.

| Read path | Latency |
| --------- | ------: |
| `Check`, one object | 1.4 ms |
| `Check`, 300 documents | ~410 ms |
| `ListObjects`, all at once | **1.9 ms** |

`ListObjects` is **220× faster** than checking each document, so the allowed
set can be fetched *before* the search and handed to the index as a filter.
The service stays authoritative and retrieval stays pre-filtered — the
recommended pattern is not the right one here.

| Retriever | vs ceiling | Local | Global | Throughput |
| --------- | ---------: | ----: | -----: | ---------: |
| SQL predicate pre-filter | 86.8% | 94.6% | 32.9% | 217 q/s |
| **OpenFGA pre-filter** | **86.8%** | **94.6%** | **32.9%** | 34 q/s |
| post-filter (the usual advice) | 84.5% | 92.6% | 30.4% | 446 q/s |

Identical quality. The cost is throughput: 6.4× slower, because every query
passes a 300-element allowed-set array to Postgres. The `ListObjects` call
itself is cheap and happens once per request; the array is the expense, and
a production system would keep it in a session-scoped temporary table.

**Where this stops working:** `ListObjects` returns a list, so its cost grows
with what the user can see. At 300 documents per tenant it is an easy win. At
a million it is not, and the architecture has to change. The pattern that is
obviously right here is not right at every size.

#### Revocation

| | |
| --- | ---: |
| `eng@gitlab` before | 217 documents |
| after removing group membership | **22 documents** |
| after restoring it | 217 documents |
| propagation | **3.0 ms** |

Immediate, because access is computed from tuples at read time rather than
materialised into an index that would need rebuilding. The stale-permission
window is one round trip — which is the argument for the service, and the
thing the SQL predicate cannot offer without a re-index.

### Phase 6 — leaks a permission check cannot see

Graph retrieval reports a **leak rate of 0.0%**. Every returned document is
one the user may read. By every metric built so far, the system is clean.

It is not. Filtering the *destination* of a traversal says nothing about
what the traversal walked *through*, and a path is information even when its
endpoint is permitted.

| Persona | Paths | Transit violations | Inferable documents |
| ------- | ----: | -----------------: | ------------------: |
| exec (all three tenants) | 11,678 | **0.0%** | 0 |
| eng@sourcegraph | 2,370 | 4.4% | 59 |
| finance@gitlab | 1,423 | 11.5% | 68 |
| eng@gitlab | 2,099 | 17.5% | 161 |
| newhire@gitlab | 1,309 | 25.8% | 420 |
| **newhire@sourcegraph** | 224 | **64.3%** | 144 |
| **all** | **29,118** | **5.8%** | **1,174** |

**The gradient is the finding.** Executives have a 0% violation rate because
nothing is forbidden to them. The least privileged persona has 64.3% — two
out of every three results were selected by documents they are not allowed
to see. 35 distinct forbidden documents shaped results across the sample,
and none of it moved the leak rate by a single point.

**Concept disclosure measured 0.0%.** No concept in this graph has *every*
source document forbidden to a given persona, so the leak mode exists in
theory and does not manifest here. Reporting it as a defended risk would be
dishonest; it is an undefended one that happens not to fire on this data,
and an LLM-extracted graph or community summaries would likely change that.

#### The fix, and what it cost

Guarding checks every node on a path rather than only its destination —
`all(node IN nodes(p) ...)`. Concepts are checked by *witness*: a concept is
visible when the user can read at least one document mentioning it, because
a concept named by both an internal and a restricted document is legitimate
to traverse.

| | vs ceiling | Local | Global | Throughput |
| --- | ---------: | ----: | -----: | ---------: |
| graph, unguarded | 84.5% | 91.7% | 33.5% | 132 q/s |
| graph, guarded | **84.6%** | **91.9%** | 33.4% | **157 q/s** |

Transit violations go to zero. Recall does not move — it improves by 0.1
points — and throughput rises 19%, because pruning paths early does less
work than walking them and discarding the results.

**The fix is free.** That matters more than it sounds: it removes the only
argument for not doing it. The 64.3% of paths stripped from the lowest
privilege persona were redundant, reaching documents other paths already
reached. A sparser graph might not be so forgiving, which is exactly why
this was measured rather than assumed.

#### What remains undefended

The 1,174 inferable documents are not fixed. Degree is information: a
permitted document with six neighbours of which two are visible tells the
user four documents exist that they cannot see. This is latent rather than
exploited, because nothing in the current interface exposes neighbour
counts — but any explainability feature that shows *why* a result surfaced
would expose it immediately, and that is the feature graph systems are built
to offer.

### Phase 5 — the graph does not help, and here is why

The result the project was built to be able to detect.

| Retriever | Local | Global | Throughput |
| --------- | ----: | -----: | ---------: |
| dense pre-filter (control) | **94.6%** | 32.9% | 202 q/s |
| graph, equal weight | 91.7% | 33.5% | 132 q/s |
| graph, seed-weighted | 94.6% | 32.9% | 199 q/s |
| graph, expansion-weighted | 55.4% | 29.7% | 164 q/s |
| *control + reranking* | *91.3%* | ***42.5%*** | *0.7 q/s* |

Graph expansion moves global recall by **+0.6%** and costs 2.9 points of
local. Cross-encoder reranking — a far more ordinary technique — gains
**+10.4%** on the same queries. The graph is not close.

#### Why, specifically

A null result is only useful if the cause is known, so the graph was measured
directly rather than tuned hopefully:

| | |
| --- | ---: |
| Section's other documents reachable within two hops | **29%** |
| `LINKS_TO` edges staying inside a section | 50% |
| Shared-concept pairs staying inside a section | 57% |
| Documents with no edges at all | 130 of 639 (20%) |
| Average documents per concept | 3.0 |

Global queries here are section-shaped, and the graph does not encode section
membership. Roughly half its edges leave the section, a fifth of documents
are isolated, and even seeding traversal with a *known correct* document —
more generosity than a real query gets — reaches only 29% of the rest of its
section. No weighting fixes that. The edges are real relationships; they
point somewhere else.

#### What this does and does not show

It shows that a link-and-heading graph over this corpus does not improve
retrieval on these queries. It does **not** show that GraphRAG fails
generally. Two things remain untested: LLM-extracted entity graphs, which
produce denser and differently-shaped edges, and community summarisation,
which is the mechanism Microsoft's GraphRAG actually uses for global queries
and which needs an LLM to write the summaries. What was built here is
traversal, and traversal is not the same thing.

#### The router also failed, and the accuracy figure hides it

Since Phase 3 showed reranking is worth +10.4% on global queries at 174× the
cost, it only pays if it runs on the queries that benefit. The router
classifies from the shape of the dense score distribution — peaked for a
single-answer query, flat when the answer is spread.

| Threshold | Overall | Local | Global |
| --------: | ------: | ----: | -----: |
| 0.030 | **79.2%** | 95.8% | **6.1%** |
| 0.055 | 70.0% | 76.8% | 40.2% |
| 0.090 | 53.3% | 48.2% | 75.6% |

The best "accuracy" comes from predicting local almost always. **Always
guessing local scores 81.4% — the router is worse than the trivial
baseline.** Score spread does not separate these classes, and reporting
accuracy alone on an 81%-local dataset would have dressed up majority-class
bias as a working component.

End to end, routing recovered 0.5 of the 10.4 points available:

| | Local | Global | Throughput |
| --- | ----: | -----: | ---------: |
| dense only | 94.4% | 32.1% | 381 q/s |
| rerank everything | 91.3% | **42.5%** | 0.7 q/s |
| routed | 92.9% | 32.6% | 7.1 q/s |

A useful router needs a stronger signal than score geometry — a trained
classifier over query text, or an LLM call cheap enough to precede the
expensive stage.

### Phase 4 — the knowledge graph

| | |
| --- | ---: |
| Documents | 639 |
| Concepts | 297 |
| Relationships | 1,655 |
| Build time | 0.9s |
| **Extraction cost** | **$0.00** |
| Cross-tenant relationships | **0** |

**No LLM was used, and that was a measurement rather than a budget
constraint.** Handbooks already contain an authored graph: writers link one
page to another because the pages are related, and headings name the concepts
a page is about. Both are exact — a link either exists or it does not — and
free. Paying a model to infer relationships that were explicitly written down
is worth doing only after the written-down ones are exhausted.

The limitation is real and named rather than hidden: this recovers
document-to-document relationships and the concepts documents mention. It
does not recover entity-to-entity facts of the "Person A reports to Person B"
kind, which is what LLM extraction adds. The extractor interface is narrow so
an LLM implementation can be dropped in and the difference measured.

Link resolution runs at 17–39% because the corpus is a sample — a link only
becomes an edge when both endpoints were ingested, so edges fall roughly with
the square of the sampling rate. Concept nodes carry the connectivity that
links alone cannot.

**Cross-tenant entity resolution is prevented structurally, not by a check.**
A concept's identity is `tenant::name`, so "compensation review" in two
tenants are two different nodes that cannot be merged. A check can be
forgotten; an identifier that makes the mistake unrepresentable cannot.

#### The leak surface Phase 6 has to defend

| | |
| --- | ---: |
| Concepts naming confidential or restricted material | 149 of 297 |
| Concepts spanning more than one sensitivity tier | **109 (37%)** |
| Document pairs reachable through a shared concept | 814 |

That middle row is the problem stated as a number. A concept mentioned by
both an internal document and a restricted one is a path from material a user
may read to material they may not — and its *name alone* can disclose that
the restricted document exists and what it concerns. No document-level
permission check catches this, because nothing about the query touched a
forbidden document.

### Phase 3 — the frozen control group

Everything the knowledge graph must beat. Full 884-case set unless noted.

| Retriever | vs ceiling | Local | Global | Throughput |
| --------- | ---------: | -----: | -----: | ---------: |
| dense pre-filter | **86.8%** | **94.6%** | 32.9% | 243 q/s |
| + lexical fusion (RRF, weight 0.25) | 84.3% | 91.1% | 35.0% | 104 q/s |
| + cross-encoder rerank *(200-case sample)* | 85.6% | 91.3% | **42.5%** | 0.7 q/s |

**Two independent techniques, one identical pattern.** Lexical fusion and
cross-encoder reranking were added separately, and both improved
cross-document questions while damaging point lookups:

| Technique | Local | Global |
| --------- | ----: | -----: |
| Lexical fusion | −3.5% | +2.0% |
| Cross-encoder rerank | −3.2% | **+10.4%** |

That convergence is the finding. A point lookup has one right answer and the
dense ranking usually already found it, so anything that reshuffles the top
results mostly introduces noise. A cross-document question needs several
documents that share a term or a theme rather than a single best match, and
both techniques are good at exactly that.

**There is no single configuration to freeze, and that is the result.**
Optimising for the average would give up 3 points on the queries dense search
already handles well in exchange for gains on queries it handles badly — or
the reverse. Routing by query class is not a refinement here; it is what the
measurements say to do. Phase 5 reaches the same conclusion about the
knowledge graph from a completely different direction.

**Reranking costs 174× throughput** — 122 down to 0.7 queries per second on
CPU, because a cross-encoder runs a forward pass per candidate and can
precompute nothing. A +10.4% gain on global queries is worth that on a
routed path and absurd on every query.

**The bar for the knowledge graph is now 42.5% on global queries**, not the
25.0% Phase 1 produced. Ordinary techniques closed most of that gap already,
which is exactly the comparison a graph is usually never measured against.

### What Phase 2 found

**Both filter placements reach a zero leak rate. They are not equivalent.**
Pre-filtering beats post-filtering on every quality measure — 86.8% versus
84.5% of ceiling, and a wider margin on global queries. Same policy, same
results budget; the only difference is whether the database knew who was
asking before it chose the candidates.

**The SQL policy is provably identical to the reference implementation** —
7,668 persona-document comparisons, zero disagreements, checked in CI. This
is the payoff for keeping `access.py` deliberately slow and obvious: the
optimised path is verified equivalent rather than assumed to be.

**I predicted post-filtering would collapse recall, and at the default
settings it did not.** Post-filtering scored *better* than no filtering at
all. The prediction was directionally right and mechanically wrong, so the
question got measured properly instead of quietly dropped:

| Oversample | Post-filter, new hire | Post-filter, exec | Pre-filter, new hire | Gap |
| ---------: | --------------------: | ----------------: | -------------------: | --: |
| 1 | 82.1% | 93.2% | 96.4% | 14.3% |
| 6 | 85.7% | 94.0% | 100.0% | 14.3% |
| 25 | 85.7% | 94.0% | 100.0% | 14.3% |
| 50 | 92.9% | 94.0% | 100.0% | 7.1% |
| 100 | 96.4% | 94.0% | 100.0% | 3.6% |
| 200 | 96.4% | 94.0% | 100.0% | 3.6% |

The collapse is real, and worse than a simple tuning problem. For a persona
entitled to 7 of 114 documents, post-filtering plateaus at 85.7% and stays
there until **50× oversample** — and even fetching 1,000 chunks to return 10
documents never quite reaches what pre-filtering achieves at 6×. The
workaround costs more than the fix.

**The executive's number never moves.** It sits at 94.0% from 6× onward,
regardless of everything happening to the new hire. That is precisely why
this failure survives code review: whoever builds and tests the system is
almost never the person it happens to.

**Pre-filtering costs roughly half the throughput** — 199 versus 409 queries
per second.

*Corrected later:* the explanation given here at the time was that a
selective `WHERE` degrades the HNSW index. Query plans in the Qdrant
benchmark below show something different — Postgres abandons the HNSW index
entirely under a filter and runs exact search over the permitted rows. The
cost is the planner changing strategy, not an index performing badly, and
the exact search it switches to returns 100% recall.

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
