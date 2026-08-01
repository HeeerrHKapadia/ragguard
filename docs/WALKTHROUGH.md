# ragguard — project walkthrough

A guided tour of what has been built, in the order it was built, with the
reasoning behind each decision. Written to be readable by someone who has
never seen the codebase.

**Repository:** https://github.com/HeeerrHKapadia/ragguard
**Status as of this writing:** 4 commits, 2,463 lines, 32 tests, 884 graded
evaluation cases. No retrieval code exists yet — deliberately.

---

## 1. What this project is

An enterprise knowledge engine where **who is asking changes what can be
retrieved**.

Most RAG systems answer one question: *what is most relevant?* This one has to
answer a harder one: *what is most relevant that this person is allowed to
see?* — and prove it, with numbers, under adversarial pressure.

Three real company handbooks stand in for three tenants. Each tenant has
personas ranging from a brand-new hire to an executive. The same question asked
by different people must return different documents, and the system has to be
measurably correct about that.

### Why this problem is worth solving

Relevance alone leaks. A query from an intern and a query from the CFO produce
the same nearest neighbours in embedding space; only an authorization layer
separates them. The standard failure modes are all real and all measurable:

| Failure mode | What goes wrong |
| --- | --- |
| Post-filter recall collapse | Retrieve top-k, then drop what the user cannot see. The VP gets 10 results, the intern gets 2. Both "work". |
| ANN index degradation | Restrictive filters sever HNSW graph links. At high selectivity the graph disconnects and recall falls apart. |
| Prompt-level "enforcement" | "Only use tenant X's documents" fails under prompt injection. LLMs are probabilistic; access control cannot be. |
| Existence leakage | "I found 3 documents you cannot access" leaks that they exist. So does a suspiciously specific refusal. |
| Stale ACLs | Someone changes teams; the index still carries their old access. Revocation latency is a security property. |

The knowledge-graph layer, arriving in Phase 4, adds four more that vector RAG
simply does not have — edges being sensitive in themselves, community summaries
laundering confidential sources, multi-hop traversal escaping the ACL boundary,
and entity resolution accidentally bridging tenants.

---

## 2. The governing principle

> Build the ruler before the thing being measured.

Every phase is ordered around this. The evaluation harness was written and
calibrated **before a single line of retrieval code existed**.

The reason is not tidiness. If you build retrieval first and evaluation second,
you unconsciously design the evaluation around what your system already does
well. Building the evaluation first means the system has to meet a standard it
did not get to define.

The second consequence is that Phase 1 is deliberately a **failure**: a naive
baseline that leaks, whose bad numbers get published. Without an honest
before-picture, no later improvement can be shown to be an improvement.

---

## 3. What exists today

Four commits, each a complete phase.

### Commit 1 — `233b9af` Set up Postgres and pgvector with the multi-tenant schema

Docker Compose stack: Postgres 17 with the `pgvector` extension, a named
volume for data, health checks so Compose waits for the database to actually
accept connections rather than merely start.

The schema covers tenants, users, groups, group membership, documents, and
chunks.

**Design decision — `chunks.tenant_id` is denormalized on purpose.** It is
derivable from `document_id` through a join. But every retrieval query must
filter by tenant, and a join inside a filtered vector search is precisely what
wrecks recall and latency. *The filter column has to live on the row being
scanned.* This same principle later drives `documents.section` and, in Phase 4,
ACL metadata on every graph node and edge.

**Design decision — no HNSW index yet.** Without it Postgres does exact
nearest-neighbour search: perfect recall, fine at this scale. It gets added in
Phase 1 so the cost of approximation is *measured* rather than assumed, and
again in Phase 3 to measure what permission filters cost on top. Creating it
early would hide both lessons behind "it just worked".

**The smoke test is the deliverable.** It does not merely check that rows come
back — it inserts vectors with *known geometry* and asserts the resulting
cosine distances are `0.0`, `0.2929`, and `1.0`. A wrong ranking fails the
test. It runs inside a transaction that is rolled back, so it is repeatable.

The final assertion is the whole project in miniature: two tenants hold chunks
with *identical* embeddings, so an unfiltered query returns another tenant's
confidential document as a perfect match. Relevance alone leaks — demonstrated
before any RAG code exists.

### Commit 2 — `555432a` Build the corpus and identity model

Three real public handbooks become three tenants:

| Tenant | Source | Documents |
| --- | --- | ---: |
| GitLab | `gitlab-com/content-sites/handbook` | 300 |
| PostHog | `PostHog/posthog.com` | 225 |
| Sourcegraph | `sourcegraph/handbook` | 114 |

Real documents matter. A synthetic corpus would make retrieval look better than
it is, because invented documents separate cleanly in a way real ones never do.

**The insight that makes the policy defensible.** Real handbooks already
contain sections named `board-meetings`, `acquisitions`, `ceo`, `finance`,
`legal`, `engineering`. That is a naturally occurring sensitivity gradient. So
tiers are *derived* from directory structure rather than hand-assigned:

```
board-meetings, acquisitions, ceo, leadership  ->  restricted
finance, legal, hiring, people-group           ->  confidential
engineering, marketing, support                ->  internal
company, values, story                         ->  public
```

When asked "how did you decide what is confidential?", the answer is
structural. These are boundaries the real companies drew.

**A problem the dry-run caught.** `corpus_report.py` inspects the tier
distribution *before* anything is written to the database. It revealed that
`restricted` was only 2% of the corpus. Investigation showed the source data
was the constraint, not the code — GitLab publishes exactly **one**
board-meeting document, for obvious reasons.

The fix was in the sampler, not the policy: `never_sample_tiers: [restricted]`.
Those documents are the entire basis of every future leak test, so the bulk
`internal` tier absorbs the trimming instead. Restricted went from 2% to 8%.

**`access.py` is deliberately the slow, obvious version.** Later phases push
enforcement into SQL filters and an authorization service for speed. The only
way to know a fast path is correct is to keep a reference implementation that
is obviously correct and diff against it. When the evaluation reports a leak,
this is the oracle that says what should have happened.

**Tenant isolation is checked first and nothing overrides it.** It is tempting
to model "exec" as simply high-ranking and compare clearance ranks everywhere.
That collapses two different questions into one. An exec at GitLab is not
*slightly* authorized to read PostHog's material; they are entirely
unauthorized. That collapse is how cross-tenant leaks get written.

### Commit 3 — `36085af` Bump CI actions

Housekeeping — cleared a Node 20 deprecation warning.

### Commit 4 — `eef9572` Build the eval harness before any retrieval exists

884 graded cases from 220 distinct queries.

**Two kinds of ground truth, from very different places.**

*Security truth is free and exact.* Whether a persona may read a document is
decided by `access.py`. No labelling, no ambiguity — a leak is a leak. This is
the unusual luxury of a permission-aware system, and it is why leak rate can be
a **hard CI gate** rather than a guideline.

*Relevance truth is the expensive kind.* Knowing which documents answer a
question normally requires human annotation. The standard workaround inverts
the problem: instead of writing a question and hunting for its answer, take a
document and derive a question it answers. The source document is then relevant
by construction.

The bias is documented honestly — title-derived queries share vocabulary with
their source, which flatters lexical search. That is acceptable because every
retriever faces the same queries, so *comparisons* stay fair even if absolute
numbers run optimistic.

**Queries are split into two classes**, because the entire GraphRAG thesis
rests on the distinction:

- **local** — answerable from one document. Plain hybrid search should win.
- **global** — requires connecting several. The only place a knowledge graph
  can justify its 3–10× token cost.

Without this split there is no way to demonstrate the graph earns its keep.

---

## 4. The metrics

| Metric | Definition | Target |
| --- | --- | --- |
| **Leak rate** | Fraction of cases returning any forbidden document | Exactly 0 |
| **Recall vs ceiling** | Recall as a fraction of what is achievable at this `k` | Higher |
| **Recall parity** | Lowest-privilege recall ÷ highest-privilege recall | ≈ 1.0 |
| **Over-block rate** | Entitled material not returned | Lower |

**Leak rate is case-level, not document-level, and deliberately so.** A system
returning one forbidden document among ninety-nine correct ones has still
leaked. Averaging would report a soothing 1% and bury the breach.

**Recall parity is the metric this project exists to expose.** The failure it
catches is specific and silent: retrieve top-k, drop what the user may not see,
and a low-privilege persona ends up with a handful of results while an exec
gets a full page. Both look like the system "worked". Leak rate stays at zero,
faithfulness stays high, and nothing in a conventional RAG scorecard moves —
but the new hire is quietly getting a much worse product.

---

## 5. Calibrating the harness

An evaluation harness is itself software, and software that grades other
software is unusually easy to get wrong in a way nobody notices. A scorer with
an inverted comparison reports encouraging numbers forever.

So three reference retrievers bracket the measurement space, and CI asserts
their scores:

| Retriever | Behaviour | Leak rate | Recall / ceiling |
| --- | --- | ---: | ---: |
| `null` | Returns nothing | 0.0% | 0% |
| `leaky` | No filtering at all | **82.5%** | 100% |
| `oracle` | Exactly what is allowed and relevant | 0.0% | 100% |

`null` is kept permanently as a reminder that leak rate alone is not a success
criterion — returning nothing scores a perfect zero leaks and is useless.

**`oracle` scores recall parity 1.00, which proves parity is reachable.** When
naive post-filtering drops it in Phase 2, that is a real regression rather than
a law of nature. Without this baseline the difference is invisible.

### Three bugs the calibration found

Writing retrievers with known-correct scores immediately exposed three defects
in my own code. None would have been visible from a real retriever's numbers.

| Bug | Symptom | Root cause |
| --- | --- | --- |
| `LeakyRetriever` scored 32%, not 82% | Looked "mostly safe" | It only returned *relevant* documents, so it leaked only when a relevant document happened to be forbidden. A genuinely unfiltered system returns whatever ranks high across everything it indexed. |
| The oracle "failed" at 95.3% recall | A perfect system scoring imperfect | **recall@k is capped by k.** A global query with 25 relevant documents cannot exceed 40% recall at `k=10`. My expectation of 1.0 was arithmetically impossible. |
| Goldens differed between runs | Determinism check failed | `Counter.most_common` breaks ties by insertion order; insertion order came from iterating a **set**; Python randomizes string hashes per process. Tied words emerged in different orders, changing the generated query text. |

The second one is the most instructive. The correct response was **not** to
loosen the failing threshold — it was to add `recall_vs_ceiling`, which
separates the `k` cutoff from genuine retrieval misses. Raw recall punishes a
retriever for a limit it did not choose.

> When a test fails against a system you know is perfect, suspect the metric,
> not the threshold.

The third was fixed by sorting on `(-count, word)` so tie-breaking never
depends on hash order, verified identical across three runs with
`PYTHONHASHSEED=random`.

---

## 6. Running it

```bash
cp .env.example .env
```

```bash
docker compose up -d --wait
```

```bash
uv sync
```

```bash
uv run python scripts/smoke_test.py
```

```bash
uv run python scripts/fetch_corpus.py
```

```bash
uv run python scripts/seed.py
```

```bash
uv run python scripts/access_matrix.py
```

```bash
uv run python scripts/build_goldens.py
```

```bash
uv run python scripts/evaluate.py
```

### What each script is for

| Script | Purpose |
| --- | --- |
| `smoke_test.py` | Proves the database works, including similarity *ranking* |
| `fetch_corpus.py` | Sparse, blobless, depth-1 clones of three handbooks |
| `corpus_report.py` | Dry-run ingestion; inspect tier distribution before writing |
| `seed.py` | Config to database rows; idempotent upserts |
| `access_matrix.py` | Who can see what — the visibility matrix |
| `build_goldens.py` | Generate the 884-case graded dataset |
| `evaluate.py` | Score retrievers; calibrate the harness |

---

## 7. Current status

**CI is red.** The `Golden dataset is deterministic` step fails on Linux. The
hash-ordering fix verified locally on Windows did not fully hold in CI, meaning
there is a second source of non-determinism still to find. Everything before
that step passes: lint, 32 tests, smoke test against a live database, corpus
fetch, seed, and the policy sanity check.

This is exactly the check doing its job. A determinism guarantee that only held
on one machine would have been worse than none, because it would have been
believed.

---

## 8. What comes next

| Phase | Work |
| --- | --- |
| **1** | Naive baseline that leaks. Fixed chunks, dense-only, prompt-level "enforcement". Publish the bad numbers. |
| **2** | Move enforcement into the retrieval layer. Drive leak rate to zero, discover post-filter recall collapse. |
| **3** | Hybrid + reranking — **the control group** the graph must beat. Freeze these numbers. |
| **4** | Knowledge graph construction. LLM extraction into Neo4j, ACL metadata on every node and edge, per-tenant entity resolution. |
| **5** | Graph retrieval with local/global routing, head-to-head against Phase 3. |
| **6** | Permission-aware traversal — the novel part. Edge sensitivity, summary laundering, hop-boundary enforcement. |
| **7** | OpenFGA authorization across both stores. |
| **8** | Adversarial red-team suite, wired into CI as regression tests. |
| **9** | Ship — API, tracing, CI gates, persona and graph-visualization demo. |

The thesis the whole project is built to test:

> Building GraphRAG is not impressive. Proving *where* it beats hybrid search
> on your corpus — and where it loses — is.
