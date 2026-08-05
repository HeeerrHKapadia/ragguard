---
title: ragguard
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ragguard

Permission-aware retrieval. The same question asked by different people
returns different documents, and the difference is enforced in the database
rather than requested of a model.

Pick a persona, search, and watch the results change:

| Persona | What they get for "compensation review and pay bands" |
| ------- | ----------------------------------------------------- |
| `newhire@gitlab.test` | Public pages only. Nothing about compensation |
| `eng@gitlab.test` | Internal compensation policy |
| `exec@gitlab.test` | Restricted review conversations, confidential assessments |

Zero of eight results are shared between the new hire and the executive.

The corpus is three real public company handbooks — GitLab, Sourcegraph and
PostHog — treated as separate tenants, with sensitivity tiers derived from
the directories those companies already use.

**This demo runs on 240 documents; the measured results in the repository
use 639.** Embedding dominates the image build and a free Space build has a
time limit, so the corpus is trimmed here.

Restricted documents are exempt from the trim, so all 52 survive and the
executive still sees material an engineer cannot. Every persona sees a
different corpus at this size — that property is checked rather than assumed
— though a new hire has fewer public pages to find than in the full run.

Everything runs locally inside the container: Postgres with pgvector, a
384-dimension ONNX embedding model, and the API. No external service, no API
key, no network calls at request time.

Source, measurements, and the parts that did not work:
**https://github.com/HeeerrHKapadia/ragguard**
