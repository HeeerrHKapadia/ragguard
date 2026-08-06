# ragguard — Complete Study & Recruiter Interview Prep Guide

> A single document to (1) understand the project from scratch, (2) know exactly what
> exists in the codebase, (3) answer any recruiter question, and (4) survive the hard
> "grill" follow-ups. Numbers here come from the repo's own README, code, and eval output.

---

# PART 0 — The 60-second pitch (memorize this)

> "ragguard is a **permission-aware GraphRAG knowledge engine**. Normal RAG answers
> *'what's most relevant?'* — ragguard answers *'what's most relevant **that this person is
> allowed to see**?'*, across both vector search and a knowledge graph, for multiple
> tenants and roles. The hard part isn't building retrieval; it's proving retrieval doesn't
> **leak**. So I built reference baselines, defined a **leak-rate** metric, and measured every
> retriever against 884 graded cases over 639 real documents and 12 personas. I even
> red-teamed it with adversarial attacks that are now permanent regression tests. The
> headline finding: a naive dense retriever leaks **more** (93%) than one built to be reckless
> (82%), because semantic similarity ignores organizational boundaries. Enforcement is
> deterministic and lives **in the database, before ranking** — not in the prompt."

**Five numbers to keep on the tip of your tongue:**
- **639** documents, **4,996** embedded chunks, **12** personas, **3** tenants.
- **93.1%** leak (naive dense) vs **0.0%** (pre-filter) — the whole point.
- **25.0%** global vs **92.6%** local recall for dense — the reason the graph exists.
- **384**-dimension embeddings (BAAI/bge-small-en-v1.5, ONNX).
- Revocation propagates in **~2.4 ms** (217 → 22 visible documents).

---

# PART 1 — Understand the project from scratch

## 1.1 The problem: relevance alone leaks

A query from an intern and a query from the CFO produce the **same nearest neighbours**.
Only the authorization layer separates them. If you rely on similarity, you leak. The project
enumerates the standard, *measurable* failure modes:

| Failure mode | What goes wrong |
| --- | --- |
| **Post-filtering destroys recall** | Retrieve top-k, then drop what the user can't see → the VP gets 10 results, the intern gets 2. Both "work"; only one is useful. |
| **Pre-filtering degrades ANN indexes** | Restrictive filters sever HNSW graph links; at high selectivity the graph disconnects and recall collapses. |
| **Prompt-level enforcement isn't enforcement** | "Only use tenant X's docs" fails under prompt injection. Filtering must be deterministic, in the DB, before context reaches the model. |
| **Existence leakage** | "I found 3 documents you can't access" leaks that they exist. So does a suspiciously specific refusal. |
| **Stale ACLs** | Someone changes teams; the index still carries old access. Revocation latency is a security property. |

The **graph layer** adds four more failure modes vector RAG doesn't have:

| Graph failure mode | What goes wrong |
| --- | --- |
| **Edges are sensitive** | Knowing `Ana → works_on → Project Titan` leaks Titan exists and who staffs it, even if every document is blocked. |
| **Community summaries launder sensitivity** | A summary aggregated across many docs inherits the sensitivity of *all* of them. |
| **Multi-hop traversal escapes the ACL boundary** | Traversal walks from an allowed node *through* a forbidden one. Guarding every hop can disconnect the graph for low-privilege users. |
| **Entity resolution bridges tenants** | If extraction decides "Ana Ruiz" in tenant A and B are the same entity, the index itself is a cross-tenant leak. Resolution must be per-tenant. |

## 1.2 The guiding philosophy (this is *why it's impressive*)

- **Measured, not assumed.** The GraphRAG claim is only accepted if the graph *beats a frozen
  hybrid baseline per query class* to justify its 3–10× token cost. Where it loses, a **router**
  sends queries to the cheap path instead.
- **Reference retrievers bracket reality.** `null` (returns nothing, 0% leak, 0% recall) and
  `leaky` (no filtering, maximum leak) aren't systems you'd ship — they calibrate the harness
  so every real number falls inside a known range.
- **Adversarial by default.** Every attack that ever worked becomes a permanent regression test.

## 1.3 Architecture (components & data flow)

```
                 ┌────────────────────────────────────────────┐
   Browser  ───► │  FastAPI service (ragguard.api.app)         │
  (demo UI)      │   /api/token  → issue JWT (identity only)   │
                 │   /api/search → PreFilterRetriever          │
                 │   /api/graph  → multi-hop, ACL-guarded      │
                 │   /api/health → chunk count                 │
                 └───────┬───────────────┬───────────────┬─────┘
                         │               │               │
              pre-filtered SQL     Cypher visibility   authz checks
                         ▼               ▼               ▼
                 ┌───────────────┐ ┌───────────┐ ┌───────────────┐
                 │ Postgres 17   │ │  Neo4j 5  │ │   OpenFGA     │
                 │ + pgvector    │ │ knowledge │ │ Zanzibar-style│
                 │ vectors +     │ │  graph    │ │ authorization │
                 │ tsvector +    │ └───────────┘ └───────────────┘
                 │ ACL metadata  │
                 └───────────────┘   (Qdrant = optional comparison only)
```

**Golden rule of the design:** identity comes **only** from the verified token. Nothing
downstream accepts persona/tenant/clearance from the request body — otherwise changing what
you can read would be editing a JSON field, which is the exact bug class the project exists to
prevent.

## 1.4 The data model

- **tenants** → **groups** → **users** (a user belongs to groups; groups carry a `clearance`
  and optional `elevated_sections`).
- **documents** (per tenant, with a `sensitivity` tier) → **chunks** (text + 384-dim
  `embedding` + denormalized `section`/`sensitivity` for fast ACL filtering).
- Sensitivity tiers: **public → internal → confidential → restricted** (ordered).
- Tiers are **derived from the directory** each doc lives in (`board-meetings`/`acquisitions`
  → restricted, `finance`/`legal` → confidential, `engineering` → internal). These are
  boundaries the real companies drew, not invented for a demo.

**Corpus (639 real documents from 3 public handbooks = 3 tenants):**

| Tenant | Docs | public | internal | confidential | restricted |
| --- | --: | --: | --: | --: | --: |
| GitLab | 300 | 22 | 195 | 49 | 34 |
| PostHog | 225 | 28 | 167 | 26 | 4 |
| Sourcegraph | 114 | 7 | 67 | 26 | 14 |
| **Total** | **639** | **57** | **429** | **101** | **52** |

**Personas (GitLab example) — the intern-vs-exec spread is the problem in one line:**

| Persona | Clearance | Visible (of 300) |
| --- | --- | --: |
| `newhire@gitlab` | public | 22 |
| `eng@gitlab` | internal | 217 |
| `finance@gitlab` | internal + finance elevation | 230 |
| `exec@gitlab` | restricted | 300 |

The **finance** persona is the subtle one: it reads **13 of 49** confidential docs — only the
ones under `finance/`. Real access is rarely a single global level; modeling it as one is how
systems over-block or over-share.

## 1.5 Retrieval strategies (and the key result for each)

- **Dense (no authz)** — baseline. Great at point lookups (**92.6% local**), poor at
  cross-document questions (**25.0% global**). Leaks **93.1%**.
- **Post-filter** — retrieve top-k then drop forbidden. Leak **0%**, but recall for
  low-privilege users suffers (fewer results survive).
- **Pre-filter** — never consider forbidden rows. Leak **0%**, best local/global recall, lower
  throughput (199 q/s vs post-filter 409 q/s). **This is the production path.**
- **Hybrid (RRF)** — dense + lexical (`tsvector`) fused with Reciprocal Rank Fusion.
- **Graph (GraphRAG)** — for global/cross-document questions the router deems worth the cost.

**The pgvector vs Qdrant investigation (a highlight — it corrected the author's own claim):**
- Under a permission filter, **Postgres abandons the HNSW index** and does an exact scan over
  the *permitted* rows. That means recall is 100% (exact), and a *more selective* filter is
  *faster* (fewer rows to sort). Phase 2's earlier "pre-filtering halves throughput" was the
  planner switching strategies, not the index performing badly.
- So at this corpus size (4,996 vectors) **HNSW contributes nothing to the production path**,
  because every real query carries a permission filter and none use the index.
- The first Qdrant benchmark was **unfair**: pgvector answered on a local socket while Qdrant
  paid HTTP framing per request. Switching Qdrant to **gRPC** moved narrow-filter latency from
  **9.2 ms → 1.1 ms**. The crossover where Qdrant wins is ~**250k vectors** with filters — so
  Qdrant is kept as a documented, benchmarked option, not shipped.

## 1.6 The knowledge graph layer

- Built in Neo4j: **Documents**, **Concepts**, and relationships (`LINKS_TO`, `MENTIONS`).
- Extraction is deterministic (**$0.00, no LLM calls**) — 639 docs, 297 concepts, 1,655
  relationships.
- **Tenant isolation is a property, not a database:** Neo4j Community allows one DB, so tenants
  are separated by a property on *every node and relationship* — isolation must hold inside
  every query, which is a deliberately harder constraint to design against.
- **Leak surface measured:** 149 concepts name confidential/restricted material; 109 span more
  than one tier (a path from readable to forbidden); 814 doc pairs reachable via a shared
  concept. The `/api/graph` endpoint enforces visibility at **every hop** and deliberately does
  **not** report how many neighbours were withheld (that count is itself an existence leak).

## 1.7 Authorization (OpenFGA / Zanzibar)

- Phase 7 replaced three hand-written policy implementations with **one** OpenFGA service that
  owns the decision (the model Google uses for Drive/YouTube permissions).
- **In-memory datastore on purpose:** tuples are rebuilt from Postgres on every run, so
  durability would create a second source of truth able to drift from the first.
- **Revocation is immediate:** removing a group membership drops `eng@gitlab` from **217 → 22**
  visible docs; restoring returns it to 217; propagation is **~2.4 ms** — because access is
  computed from tuples at **read time**, not materialized into a rebuildable index.
- `ListObjects` (what can this user see?) is **218× faster** than checking each document
  individually.
- **Policy parity is enforced in CI:** the SQL policy, the Cypher policy, and the OpenFGA
  policy must each admit **exactly** what the reference implementation admits, or the answer a
  user gets depends on which code path ran.

## 1.8 Evaluation methodology (the part that separates this from a toy)

- **Leak rate** — fraction of returned documents the persona was not permitted to see. Primary
  metric. `leaky` ≈ 82.5%, `oracle`/`null` = 0%.
- **Local vs global queries** — point lookups vs cross-document synthesis. Dense is good at
  local, bad at global; that gap is the GraphRAG justification.
- **Recall parity** — recall relative to the `oracle` (perfect + permitted). Measured against an
  exact sequential scan over the permitted set, so 100% means nothing was missed.
- **Over-block rate** — entitled material *not* returned (honestly caveated: it conflates
  filtering with ordinary ranking misses; Phase 2 separates them by running the identical
  retriever with and without filtering).
- **Golden dataset determinism** — `eval/goldens.jsonl` is committed; rebuilding must reproduce
  it **byte-for-byte** (`git diff --exit-code`), or results across phases stop being comparable.
- **Scale:** 884 graded cases from 220 distinct queries, 639 documents, 12 personas.

## 1.9 The red team (Phase 8) — attacks that became regression tests

| ID | Attack | Result |
| --- | --- | --- |
| A1 | Cross-tenant extraction (120 crafted queries) | 0 foreign documents |
| A2 | Tier escalation by exact title (101 searches) | 0 hits |
| **A3** | **Existence disclosure** | **known-open**: low-privilege returns fewer results than exec — the *count* signals hidden volume |
| A4 | Stale permissions | 217 → 22 on revoke, restored cleanly |
| A5 | Graph transit (multi-hop) | guard holds (529 paths would leak unguarded) |
| A6 | Concept tenant bridging | no concept/edge spans tenants |
| A7 | Identifier collision / sanitiser injectivity | 639 ids distinct; 7 adversarial inputs distinct |
| A8 | Document dominance | top doc appears in only 15% of result sets |

**A3 is a documented, accepted residual risk** — being honest about a known-open item is a
*strength* to present, not a weakness to hide.

## 1.10 The API and the demo

- Endpoints: `/api/personas`, `/api/token`, `/api/search`, `/api/graph`, `/api/traces`,
  `/api/health`.
- **Demo auth is intentionally demo auth:** `issue_token` hands a JWT to any seeded persona
  with no password, because the point is switching personas to watch results change. A real
  deployment swaps this endpoint for an IdP and keeps everything below it unchanged — the token
  is verified the same way either way.
- **One uvicorn worker on purpose:** the process holds a single DB connection and an in-memory
  trace buffer, neither shared across workers. Scaling out means more machines, not more
  workers in one.
- **The demo itself:** one query — *"compensation review and pay bands"* — asked by three
  personas returns three disjoint result sets (**0 of 8 results shared** between new hire and
  exec). The search path is the **same pre-filtered retriever the evaluation measured**, so the
  demo can't drift from the README's numbers.

## 1.11 Tech stack & why each choice

| Layer | Choice | Why |
| --- | --- | --- |
| Vector + relational + lexical | **Postgres 17 + pgvector** (+ `tsvector`) | One database for vectors, BM25-ish lexical search, and ACL metadata |
| Graph | **Neo4j 5 Community** | Knowledge graph with per-node/per-edge ACL metadata |
| Authorization | **OpenFGA** | Zanzibar-style relationship-based access control |
| Embeddings | **fastembed (ONNX), BAAI/bge-small-en-v1.5, 384-dim** | Inference-only; avoids pulling multi-GB PyTorch just to run a model |
| API | **FastAPI + uvicorn** | Typed, async, dependency-injection for auth |
| Runtime | **Python 3.13, `uv`** | Fast, reproducible dependency management via committed `uv.lock` |
| Comparison (optional) | **Qdrant** | Filterable HNSW, benchmarked for the scale crossover, not shipped |

## 1.12 Deployment model

- **One self-contained Docker image** bakes Postgres+pgvector, all embedded chunks, the ONNX
  model, the API, and the UI. Build ≈ 15 min; container starts in seconds with data already
  present; nothing reaches the network at request time.
- **Why "pay at build, not at start":** it's a *fixed demonstration over a pinned corpus*, so
  paying once at build beats provisioning a DB and loading data on every start. This also makes
  it free to host (GHCR for the image, GitHub Actions build minutes, Hugging Face Spaces free
  tier — zero credentials).
- The published image is **verified before it's tagged**: the workflow starts it, waits for
  `/api/health`, and asserts the corpus is actually there (an image that boots but holds no
  documents is worse than a failed build).

## 1.13 Phase map (how the project tells its story)

`0a` infra (Postgres+pgvector) → `0b` corpus → `1` dense, no authz → `2` filtering
(post/pre) → `3` hybrid (dense+lexical) → `4` graph build → `5` graph retrieval → `6` graph
leaks → `7` OpenFGA → `8` existence disclosure / red team → `9` the service (API + UI).
Each phase ships only when a **failing metric** justifies the next piece — nothing is added on
faith.

---

# PART 2 — What was actually built to run it (the environment setup)

This is what *I* (the setup process) did so you can run/demo it and explain the ops story.

**Toolchain installed:** `uv` (auto-provisions Python 3.13) and Docker Engine 29.7.2.
The Cloud Agent VM is nested, so Docker was configured with the **`fuse-overlayfs`** storage
driver (kernel overlayfs won't mount in the nested VM) and `dockerd` is started manually
(no systemd).

**Services run:** `docker compose up -d` brings up Postgres/pgvector, Neo4j, OpenFGA
(required) and Qdrant (optional). Health is polled per-service.

**Data pipeline executed end-to-end:** `fetch_corpus.py` → `seed.py` → `index_corpus.py`
(4,996 chunks embedded, ~400 s) → `build_graph.py` → `load_authz.py` (1,324 tuples).

**Verification run:** `ruff`, `pytest` (80 tests), `smoke_test.py`, `access_matrix.py`,
`verify_policy_parity.py`, `authz_benchmark.py`, `build_goldens.py` (+ byte-for-byte diff),
`evaluate.py`, `redteam.py`, `api_smoke.py` — all green.

**The "hello world":** started the API (`uvicorn ragguard.api.app:app --port 8000`) and ran
the permission-aware demo query through both the API and the browser UI, reproducing the
README's exact result split.

**Reproducibility (Cloud environment):** snapshotted the working VM → built a prebuilt
environment from it → verified in a fresh Cloud Agent → proposed `install`/`start` scripts.
Two design facts drove the scripts:
- `install` **guards the ~400 s indexing** on the DB chunk count, so re-runs finish in ~1 s
  (idempotent).
- `start` **reloads OpenFGA tuples every boot** (in-memory datastore forgets them) and runs the
  API in the foreground.
- Qdrant is **non-gating** (optional comparison service; not in the retrieval path; never
  started by CI) — a Qdrant hiccup can't break the environment.

---

# PART 3 — Recruiter Q&A bank (grouped by theme)

> For each: a crisp answer you can say out loud, tied to what exists in the repo.

## 3.1 Problem framing & motivation

**Q: What problem does this solve?**
A: Enterprise RAG where *who is asking* changes what can be retrieved. Relevance alone leaks
because an intern's and an exec's queries return the same nearest neighbours; only
authorization separates them. ragguard enforces permissions deterministically and *proves*
non-leakage with metrics.

**Q: Why is this hard / non-trivial?**
A: The naive fixes each fail measurably — post-filtering wrecks recall for low-privilege users,
pre-filtering can degrade ANN indexes, and prompt-level rules break under injection. Plus a
graph adds four new leak modes (sensitive edges, laundering summaries, multi-hop escape,
cross-tenant entity resolution).

**Q: Who's the user?**
A: Any multi-tenant, role-aware enterprise knowledge system — think an internal assistant over
Confluence/Drive where a new hire and the CFO must get *different* answers to the same question.

## 3.2 Architecture & design decisions

**Q: Walk me through the architecture.**
A: FastAPI front door → identity from a verified JWT only → `PreFilterRetriever` runs a
permission-filtered vector+lexical query in Postgres/pgvector; a graph endpoint traverses Neo4j
with an ACL guard at every hop; OpenFGA owns authorization decisions. Qdrant is a benchmarked
alternative, not in the path.

**Q: Why put everything in Postgres instead of a dedicated vector DB?**
A: One store handles vectors (pgvector), lexical search (`tsvector`), *and* ACL metadata, so a
permission filter and a similarity search happen in the same query with transactional
consistency. At this scale (≈5k vectors) it's faster and simpler; I benchmarked Qdrant and
found it only wins above ~250k filtered vectors.

**Q: Where is enforcement done, and why there?**
A: In the database, before ranking. Prompt-level enforcement isn't enforcement — it fails under
injection. Identity is never taken from the request body, only from the verified token, so you
can't escalate by editing a field.

**Q: Why one uvicorn worker?**
A: The process holds a single DB connection and an in-memory trace buffer, neither shared across
workers. Correct scaling is more machines, not more workers in one process.

## 3.3 Vectors, embeddings, retrieval

**Q: Which embedding model and why 384 dimensions?**
A: BAAI/bge-small-en-v1.5 via fastembed/ONNX — inference-only, so I avoid pulling multi-GB
PyTorch. 384-dim is the model's native size; it's a good quality/speed/storage trade-off for
this corpus.

**Q: Post-filter vs pre-filter — what's the difference and which did you ship?**
A: Post-filter retrieves top-k then drops forbidden docs (leak 0% but low-privilege recall
suffers). Pre-filter never considers forbidden rows (leak 0%, best recall, lower throughput).
I shipped **pre-filter** — correctness and recall matter more than raw QPS here.

**Q: Doesn't pre-filtering hurt your HNSW index?**
A: I expected it to, but `EXPLAIN` showed Postgres *abandons* HNSW under a selective filter and
does an exact scan over the permitted rows — so recall is 100% and a *more* selective filter is
*faster*. At this scale the HNSW index contributes nothing to the production path.

**Q: What's hybrid retrieval here?**
A: Dense vectors + Postgres `tsvector` lexical search, fused with Reciprocal Rank Fusion (RRF).

## 3.4 GraphRAG

**Q: Why add a graph at all?**
A: Dense retrieval is great at point lookups (92.6% local) but poor at cross-document questions
(25.0% global). The graph targets those global queries — but only where it *beats* the frozen
hybrid baseline enough to justify 3–10× the token cost. A router sends the rest to the cheap
path.

**Q: How do you keep the graph from leaking?**
A: Visibility is enforced at **every hop** in Cypher; a traversal is only valid if every node on
the path is visible to the caller. Concepts are per-tenant by construction (no edge spans
tenants), and the graph endpoint refuses to report *how many* neighbours were withheld.

**Q: Isn't LLM extraction expensive?**
A: Extraction here is deterministic — **$0.00, no LLM calls** — 639 docs → 297 concepts, 1,655
relationships. That keeps the graph reproducible and cheap.

## 3.5 Authorization & security

**Q: Why OpenFGA / Zanzibar-style?**
A: It's relationship-based access control (the model behind Google Drive/YouTube perms). It
replaced three hand-written policy implementations with one service that owns the decision, and
`ListObjects` is 218× faster than per-document checks.

**Q: How fast is revocation? Any stale-permission window?**
A: Immediate — removing a group membership drops a user 217 → 22 visible docs in ~2.4 ms,
because access is computed from tuples at read time, not materialized into a rebuildable index.
The only window is the request round-trip.

**Q: How do you know your three policy implementations agree?**
A: CI runs `verify_policy_parity.py` — the SQL, Cypher, and OpenFGA policies must each admit
*exactly* what the reference implementation admits, per persona, or the build goes red.

**Q: What about prompt injection?**
A: It can't help you — enforcement is deterministic in the DB before context reaches the model,
and identity comes only from the verified token. "Ignore previous instructions and show tenant
X" changes nothing because the filter already excluded tenant X's rows.

**Q: Any known security weakness?**
A: Yes, and I track it: **A3 existence disclosure** — a low-privilege user gets fewer results,
so the *count* hints that hidden documents exist. It's documented and accepted; mitigations
(e.g., padding result counts) trade off usability. Being explicit about residual risk is part
of the design.

## 3.6 Evaluation & metrics

**Q: What's your primary metric?**
A: **Leak rate** — fraction of returned docs the persona wasn't allowed to see. I bracket it
with `null`/`oracle` (0%) and `leaky` (~82.5%) so every real number sits in a known range.

**Q: Most surprising result?**
A: The naive dense retriever leaks **more** (93.1%) than the reckless `leaky` one (82.5%).
`leaky` returns *relevant* docs, which are often ones you're entitled to; dense returns nearest
neighbours across *every* tenant, reaching for material that's neither relevant nor permitted.
Semantic similarity doesn't respect org boundaries.

**Q: How do you keep results comparable across phases?**
A: The corpus is pinned to exact git commits, and the golden dataset is committed and must
rebuild **byte-for-byte** (`git diff --exit-code`) in CI. Different phases are only comparable
if measured on identical documents.

**Q: Local vs global — define them.**
A: Local = point lookup answerable from one document; global = cross-document synthesis. The
25% vs 92.6% gap for dense is the entire premise of the GraphRAG argument.

## 3.7 Engineering, infra, DevOps

**Q: How is it deployed?**
A: A single Docker image bakes Postgres+data+model+API+UI. Build ≈15 min, starts in seconds,
no network at request time. Hosted free on Hugging Face Spaces; image on GHCR; built by GitHub
Actions. The image is verified (health + corpus present) before it's tagged.

**Q: Why bake data at build time instead of provisioning at start?**
A: It's a fixed demo over a pinned corpus. Paying once at build beats paying on every start,
and it makes hosting free and startup instant. For a system that *changes*, I'd do the opposite
(small image + migrations + data load).

**Q: What does CI check?**
A: The same path a developer runs — lint, unit tests, then real services (no mocks): smoke
test, seed, index, graph, authz, policy parity, golden determinism, eval calibration, red team,
and a live API smoke test asserting no persona exceeds clearance through any endpoint.

**Q: How do you run it locally / in the cloud?**
A: `docker compose up -d` for services, `uv sync` for deps, then the pipeline scripts, then
`uvicorn ragguard.api.app:app --port 8000`. In the Cloud Agent environment I automated this
into idempotent `install`/`start` scripts (guarding the ~400 s indexing, reloading OpenFGA's
in-memory tuples each boot, Docker on `fuse-overlayfs`).

## 3.8 Behavioral / "tell me about the project"

**Q: What are you most proud of?**
A: I treated retrieval as a *security* problem, not just a relevance problem — I built
reference baselines, defined a leak metric, red-teamed it, and let measurements (not opinions)
decide what ships. I even corrected my *own* earlier benchmark when `EXPLAIN` showed the real
story.

**Q: What was the hardest bug/insight?**
A: Discovering that Postgres abandons the HNSW index under a permission filter — which
invalidated a "pre-filtering halves throughput" claim I'd made. The fix was understanding the
query planner, not the index.

**Q: What would you do next?**
A: Address A3 existence disclosure, push the Qdrant path for the >250k-vector regime, and add
community-summary handling with tier-aware redaction. See `docs/IMPROVEMENT_PLAN.md`.

---

# PART 4 — Grilling scenarios (hard follow-ups) and how to handle them

> These are the "let me push on that" moments. Each has the trap, the answer, and the honest
> caveat — recruiters trust candidates who name their own limitations.

**Grill: "Isn't this just a WHERE clause? Where's the hard part?"**
- Answer: The `WHERE` is the *easy* part. The hard parts are (1) proving it doesn't leak across
  many personas/tenants with a metric, (2) not destroying recall for low-privilege users, (3)
  keeping three policy engines in agreement, and (4) the graph's four extra leak modes.
- Caveat: For pure single-doc vector search, yes, it reduces to a filter — which is exactly why
  I benchmarked that path honestly rather than overclaiming.

**Grill: "Your corpus is tiny (5k vectors). Does any of this hold at scale?"**
- Answer: I explicitly measured the crossover. Below ~250k filtered vectors, pgvector's exact
  scan over permitted rows wins; above it, Qdrant's filterable HNSW wins, and I have the
  benchmark ready to re-run. The *architecture* (pre-filter, token-time authz, per-hop graph
  guard) is scale-independent; only the vector engine choice flips.
- Caveat: I haven't *run* production traffic at 1M+ vectors; the crossover is measured on
  perturbed real embeddings, not live load.

**Grill: "Pre-filtering can disconnect an HNSW graph and collapse recall. Doesn't it here?"**
- Answer: Not at this scale — Postgres abandons HNSW under the filter and scans permitted rows
  exactly, so recall is 100%. At large scale with a real ANN filter, that risk returns, which
  is precisely why the Qdrant filterable-HNSW comparison exists.

**Grill: "How do you handle the existence-disclosure leak?"**
- Answer: I acknowledge it (A3, known-open). The count of returned results signals hidden
  volume. Mitigations — fixed-size result padding, or never revealing counts — trade usability
  for secrecy. I chose to document it rather than ship a half-measure that hides the risk.

**Grill: "Prompt injection could still exfiltrate data, right?"**
- Answer: No — because enforcement is deterministic and pre-model. The LLM only ever sees rows
  already filtered by the verified identity. Injection can change wording, not the SQL filter.
  The failure would have to be in the DB policy, which is why parity is CI-enforced.

**Grill: "Your auth has no passwords — that's insecure."**
- Answer: That's deliberate *demo* auth so you can switch personas to see results change.
  Everything below the token is production-shaped: the token is verified (JWT, signature,
  expiry) the same way regardless of issuer. A real deployment swaps `/api/token` for an IdP and
  nothing downstream changes.
- Caveat: The dev JWT secret is a known default; a real deployment must set `JWT_SECRET` (the
  code notes a startup check as the next addition).

**Grill: "Community summaries in GraphRAG leak. How do you prevent it?"**
- Answer: A summary inherits the max sensitivity of its sources. Today the graph endpoint
  guards per-hop visibility and doesn't serve laundering summaries; tier-aware summary redaction
  is on the roadmap. I measured the leak surface (149 tier-naming concepts, 109 tier-spanning)
  so the risk is quantified, not hand-waved.

**Grill: "How do you know your eval isn't overfit / gamed?"**
- Answer: Reference retrievers bracket the range (`null`/`oracle` at 0%, `leaky` at ~82.5%), the
  corpus is pinned to commits, and the golden set must rebuild byte-for-byte. If I accidentally
  tuned to the test, the calibration retrievers would move off their known values and CI would
  catch it.

**Grill: "Neo4j Community only has one database — isn't multi-tenant unsafe?"**
- Answer: Tenants are separated by a property on every node/edge, so isolation must hold inside
  every query — a *harder* constraint I designed against on purpose. The red team's A6 verifies
  no concept or edge spans tenants, and A1 confirms 0 foreign documents across 120 crafted
  cross-tenant queries.

**Grill: "Why not just use a managed vector DB / Pinecone / RAG framework?"**
- Answer: The project's value is the *authorization and evaluation* layer, not the vector store.
  Postgres gives vectors + lexical + ACLs + transactions in one place; a managed vector DB would
  split the ACL metadata away from the vectors and complicate consistent pre-filtering. I kept
  the store boring so the security story could be rigorous.

**Grill: "What breaks if a user changes teams mid-session?"**
- Answer: Nothing stale is served — authz is computed from tuples at read time, so the next
  request reflects the new membership within ~2.4 ms (A4 stale-permission test). There's no
  index to rebuild.

**Grill: "Your throughput drops with pre-filter (199 vs 409 q/s). Isn't that bad?"**
- Answer: It's a deliberate correctness/throughput trade. Pre-filter gives the best recall and
  0% leak; for a permission-critical system that's the right side of the trade. And the "drop"
  is partly the planner choosing exact search, which *guarantees* 100% recall.

**Grill: "Show me you didn't just get lucky — reproduce a result."**
- Answer: `uv run python scripts/build_goldens.py && git diff --exit-code eval/goldens.jsonl`
  proves the dataset is deterministic; `scripts/evaluate.py` re-derives the leak/recall numbers;
  `scripts/api_smoke.py` proves every persona stays within clearance through the live API.

---

# PART 5 — Cheat sheet (numbers, files, one-liners)

**Numbers:** 639 docs · 4,996 chunks · 384-dim · 12 personas · 3 tenants · 297 concepts ·
1,655 graph relationships · 1,324 authz tuples · 884 graded cases · 220 queries.
Leak: naive 93.1% → pre-filter 0%. Local 92.6% vs global 25.0%. Revocation ~2.4 ms.
Throughput: post-filter 409 q/s, pre-filter 199 q/s. Qdrant crossover ≈250k vectors.

**Files to be able to point at:**
- `src/ragguard/api/app.py` — endpoints; `auth.py` — token→identity.
- `src/ragguard/retrieval/dense.py` — `PreFilterRetriever` (the shipped path).
- `src/ragguard/graph/` — build, retrieve, filters (per-hop ACL guard), audit.
- `src/ragguard/authz/` — OpenFGA client + model.
- `db/init/` — extensions, schema, chunk ACL denormalization.
- `config/tenants.yaml` — the security model (corpora, tiers, groups, personas).
- `scripts/` — the whole pipeline + every benchmark + `redteam.py`.
- `.github/workflows/ci.yml` — the canonical build+verify path.
- `docs/ARCHITECTURE.md`, `docs/WALKTHROUGH.md`, `docs/IMPROVEMENT_PLAN.md`.

**One-liners to run it:**
```bash
cp .env.example .env && docker compose up -d      # services
uv sync                                            # deps (Python 3.13)
uv run python scripts/smoke_test.py                # prove the DB works
uv run python scripts/fetch_corpus.py && \
uv run python scripts/seed.py && \
uv run python scripts/index_corpus.py && \
uv run python scripts/build_graph.py && \
uv run python scripts/load_authz.py                # build everything
uv run uvicorn ragguard.api.app:app --port 8000    # then open http://localhost:8000
```

**The demo to show live:** open the UI, ask *"compensation review and pay bands"* as
`newhire@gitlab` (only public results), then `eng@gitlab` (internal), then `exec@gitlab`
(restricted/confidential) — 0 of 8 results shared between the first and last.

---

*Prepared as a study aid. Every figure traces to the repo's README, code, `config/tenants.yaml`,
or the eval scripts' output — if a recruiter asks "where's that number from?", point at the
script that produces it.*
