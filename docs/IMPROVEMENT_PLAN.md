# Significant improvement plan

A ranked roadmap to make ragguard faster, safer under load, and easier to
extend — grounded in measured bottlenecks (README / WALKTHROUGH) rather than
generic “optimize everything” advice.

Each phase is meant to be shipped and measured before the next. Changing
several variables at once is how Phase 2’s HNSW claim went wrong.

---

## Goals

| Goal | Success signal |
| --- | --- |
| Cut request-path latency | p50 / p95 search latency down without raising leak rate |
| Raise concurrency | API handles concurrent searches without serializing on one connection |
| Pay for expensive stages only when they help | Router accuracy > always-local baseline; rerank spend only on global queries |
| Keep leak rate at 0 | Eval + policy parity still green |
| Scale past ~5k chunks | Path ready for filterable HNSW / OpenFGA allow-set materialization |

---

## Phase A — Request-path efficiency (this change set)

Low risk, high leverage, no corpus rebuild required.

1. **Process-wide query embedding cache** — 220 distinct queries × many
   personas currently re-embed per retriever instance.
2. **Postgres connection pool** — replace the single shared API connection so
   search / health / hydrate no longer serialize.
3. **Fold document title into retrieval SELECT** — drop the extra hydrate
   round-trip on `/api/search`.
4. **Hybrid: one round-trip + single `websearch_to_tsquery`** — dense and
   lexical rankings in one SQL statement; compute the tsquery once.
5. **Router: text features + probe reuse** — cheap lexical cues for global
   queries; when global is chosen, rerank the probe shortlist instead of
   throwing the probe away and re-retrieving.
6. **OpenFGA allow-set via `JOIN unnest`** — avoid large `= ANY(list)` binds.
7. **Structure-aware markdown chunking** (opt-in API) — heading-aware splits
   ready for the next re-index without changing the baked demo corpus yet.

---

## Phase B — Schema-level retrieval speed

Requires a migration and re-index of chunks (or a one-shot backfill).

1. **Denormalize `section` + `sensitivity` onto `chunks`** — the same reason
   `tenant_id` is already there: the filter must live on the scanned row.
2. **`visibility_sql` against chunk columns** — drop `JOIN documents/tenants`
   from the ANN / FTS hot path; join documents only for titles when needed.
3. **Partial / covering indexes** for `(tenant_id, sensitivity)` and section
   prefix checks used by elevation.

Expected effect: fewer joins per candidate, better plans once the corpus
grows past the size where exact scan over permitted rows is free.

---

## Phase C — Make expensive stages earn their keep

1. **Train / calibrate a real query classifier** on the labelled goldens
   (text + probe features). Target: beat the 81% always-local baseline with
   enough global recall to justify rerank.
2. **Adaptive shortlist** — smaller `SHORTLIST` when the probe is already
   peaked; larger only when flat.
3. **Optional late-interaction / ColBERT-style rerank** only if cross-encoder
   remains the latency cliff after routing works.
4. **CI latency gates** — smoke a fixed query set and fail on p95 regressions
   (benches today are scripts, not gates).

---

## Phase D — Scale & authorization at size

1. **Wire Qdrant (or equivalent filterable HNSW)** when vectors approach the
   measured ~250k crossover; keep pgvector for the free demo image.
2. **Materialize OpenFGA allow-sets** into a session temp table or cached
   join key when visible docs exceed ListObjects comfort (~10k+).
3. **Async OpenFGA on the live request path** with per-request allow-set
   cache and revocation TTL as a first-class metric.
4. **Graph path**: collapse hydration into one Neo4j session; cache
   concept-witness visibility; keep Neo4j optional.

---

## Phase E — Product & security hardening

1. Close existence-disclosure on result counts / graph neighbour withholding
   (Phase 8 open risk A3).
2. Structure-aware re-index of the demo corpus; re-run goldens and publish
   the delta.
3. Real auth (not demo JWTs) for non-demo deploys.
4. Expand unit tests for pre-filter SQL, router classify/reuse, authz join
   path, and hybrid single-shot ranking — behaviour today lives mostly in
   scripts.

---

## What not to do

- Do not “optimize” `access.py` — it is the slow oracle on purpose.
- Do not put OpenFGA checks after retrieval — Phase 2 already priced that.
- Do not enable community-summary GraphRAG until ACL inheritance for
  summaries is designed and leak-audited.
- Do not chase hybrid overall recall; route by query class instead.

---

## Measurement checklist (every phase)

```text
uv run pytest
uv run python scripts/verify_policy_parity.py   # if DB up
uv run python scripts/evaluate.py --retriever dense-prefilter
# latency: scripts/*_benchmark.py for the path that changed
```

Leak rate must stay **0.0%**. Recall vs ceiling and per-persona parity are
the quality bar; throughput / p95 are the efficiency bar.
