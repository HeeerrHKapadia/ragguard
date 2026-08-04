# What this project measured, and what it got wrong

The [README](../README.md) has the results and
[ARCHITECTURE.md](ARCHITECTURE.md) has the structure. This is the part that
is usually left out: the predictions that failed, and what each one cost to
find.

---

## The governing rule

> Build the ruler before the thing being measured.

The evaluation harness was written and calibrated before a single line of
retrieval code existed. Not tidiness — if you build retrieval first, you
design the evaluation around what it already does well, and every number
afterwards flatters you.

The second consequence is that Phase 1 was built to **fail**: a naive
retriever with no authorization at all, whose bad numbers were published as
the before-picture. Without that, no later improvement is demonstrable.

---

## Four predictions that were wrong

### 1. "Post-filtering will collapse recall"

It scored *better* than no filtering at all.

Rather than drop the claim, the oversample factor was swept for the persona
who can see 7 of 114 documents:

| Oversample | Post-filter | Pre-filter |
| ---------: | ----------: | ---------: |
| 6× | 85.7% | 100.0% |
| 25× | 85.7% | 100.0% |
| 50× | 92.9% | 100.0% |
| 100× | 96.4% | 100.0% |

The collapse is real; the mechanism was wrong. Post-filtering plateaus until
**50× oversampling**, and even fetching 1,000 chunks to return 10 documents
never matches pre-filtering at 6×. It is not a tuning knob — the workaround
costs more than the fix.

The executive's number never moves. That is why this survives code review:
whoever builds and tests the system is never the person it happens to.

### 2. "Hybrid search beats dense alone"

It lost overall — and the reason mattered more than the loss.

| Technique | Local | Global |
| --------- | ----: | -----: |
| Lexical fusion | −3.5% | +2.0% |
| Cross-encoder rerank | −3.2% | **+10.4%** |

Two techniques added separately, identical pattern from both. A point lookup
has one right answer that dense search already found, so reshuffling the top
adds noise. A cross-document question needs several documents sharing a term
or theme, which is exactly what both are good at.

So there is no single configuration worth freezing, and routing by query
class is not a refinement but what the measurements demand.

### 3. "Selective filters degrade the HNSW index"

`EXPLAIN` said something sharper:

```
unfiltered   Index Scan using chunks_embedding_hnsw         12.0 ms
filtered     Bitmap Heap Scan → Index Scan → top-N heapsort  7.3 ms
```

Postgres does not degrade the index under a filter. It **abandons it** and
runs exact search over the permitted rows. That explains 100% recall, why
more selective filters are *faster*, and what the earlier "halving" actually
was — a planner changing strategy, not an index struggling.

The original claim is still in the README with the correction beside it,
rather than edited away.

### 4. "RRF rewards consensus"

A fusion test failed and the **test** was wrong. Since `1/x` is convex,
`1/(k+1) + 1/(k+3)` always exceeds `2/(k+2)`, so a document ranked first by
one ranker and third by the other beats one ranked second by both. Consensus
only wins once the disagreement is wide enough.

---

## The graph did not help

The centrepiece result, and a negative one.

| Approach | Local | Global |
| -------- | ----: | -----: |
| dense (control) | **94.6%** | 32.9% |
| graph expansion | 91.7% | 33.5% |
| cross-encoder rerank | 91.3% | **42.5%** |

Graph expansion gained **+0.6%** on exactly the queries a graph is supposed
to be for. Cross-encoder reranking — entirely ordinary — gained **+10.4%**.

A null result is only useful with a cause, so the graph was measured
directly rather than tuned hopefully:

| | |
| --- | ---: |
| Section reachable within two hops | **29%** |
| `LINKS_TO` edges staying in-section | 50% |
| Documents with no edges at all | 130 of 639 |

Global queries here are section-shaped and the graph does not encode section
membership. Even seeding traversal with a *known correct* document — more
help than a real query gets — reaches 29% of the rest. No weighting fixes
that. The edges are real relationships; they point somewhere else.

**What this does not show:** that GraphRAG fails generally. LLM-extracted
entity graphs and community summarisation are both untested here. What was
built is traversal, and traversal is not the same thing.

---

## The finding worth keeping

Graph retrieval reported a **leak rate of 0.0%**. Every returned document was
one the user could read. By every metric built up to that point, clean.

It was not. Filtering the *destination* of a traversal says nothing about
what it walked *through*:

| Persona | Transit violations |
| ------- | -----------------: |
| exec (all tenants) | **0.0%** |
| eng@gitlab | 17.5% |
| newhire@gitlab | 25.8% |
| **newhire@sourcegraph** | **64.3%** |

Two of every three results for the least privileged persona were selected by
documents they are not allowed to see. Executives score zero because nothing
is forbidden to them — the same reason nobody notices.

The fix — checking every node on a path rather than only its endpoint — drove
violations to zero, cost **no recall**, and ran **19% faster**, because
pruning early beats walking and discarding.

---

## The red team found a bug in my own code

OpenFGA object ids cannot contain a colon, so document URIs get rewritten.
That rewrite was **not injective**: `a:b.md` and `a_b.md` both became
`a_b.md`, meaning two documents sharing one authorization object and
therefore each other's permissions.

It never fired. All 639 ids happened to be distinct, so every test passed —
and still passes. That is luck, not design. Fixed by appending a digest
whenever a rewrite occurs.

One risk is recorded as **open** rather than closed: result counts differ by
privilege (28 against 40 on the same queries), which signals how much is
withheld. The obvious fix does not work — padding to a fixed size needs
something to pad with, and a persona entitled to seven documents cannot
reach ten without repeating results or returning material they cannot see.

---

## Bugs I wrote twice

`@contextmanager` lifetime. `connect().__enter__()` leaves the context
manager unreferenced, so it is garbage-collected, runs its cleanup, and
closes the connection underneath you. It cost an afternoon in Phase 0a as a
hang, and reappeared in Phase 9 as a server that would not start.

The second time it got a comment.

---

## What a reviewer should take from this

Most of the value is in the negative results. The graph lost to a
cross-encoder, four predictions were wrong, and one security fix came from
attacking my own code rather than from a checklist.

None of that would exist without building the evaluation first. A system
measured only after it works produces numbers that agree with whoever built
it.
