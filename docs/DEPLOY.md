# Deploying

Getting ragguard onto Fly.io, and what to check when it misbehaves.

The demo needs **two** services: the API and Postgres. Neo4j, OpenFGA and
Qdrant are development and benchmarking dependencies — the graph endpoint
degrades to an empty result without Neo4j rather than failing.

---

## Prerequisites

- A Fly.io account and `flyctl` installed
- Docker, for building locally before pushing anything

```bash
flyctl auth login
```

---

## First deployment

**1. Create the app without deploying yet.** `--no-deploy` matters: the
database has to exist and be populated before the API starts, or the first
health check fails against an empty schema.

```bash
flyctl launch --no-deploy --copy-config --name ragguard
```

Decline when it offers to overwrite `fly.toml` — the settings there are
chosen rather than inferred.

**2. Create Postgres and attach it.** Attaching sets `DATABASE_URL` on the
app automatically.

```bash
flyctl postgres create --name ragguard-db --initial-cluster-size 1 --vm-size shared-cpu-1x --volume-size 3
```

```bash
flyctl postgres attach ragguard-db --app ragguard
```

**3. Set the token secret.** The dev default is fine for three public
handbooks and nothing else.

```bash
flyctl secrets set JWT_SECRET="$(openssl rand -hex 32)" --app ragguard
```

**4. Point the app's Postgres variables at the attached database.** The
application reads discrete variables rather than `DATABASE_URL`, so set them
from the credentials Fly printed during `attach`:

```bash
flyctl secrets set POSTGRES_HOST=ragguard-db.internal POSTGRES_USER=ragguard POSTGRES_DB=ragguard POSTGRES_PASSWORD=<from attach output> --app ragguard
```

**5. Create the schema and load the data.** Open a tunnel to the database
and run the normal pipeline against it — the same scripts used locally, so
there is no separate production path that could drift.

```bash
flyctl proxy 15432:5432 --app ragguard-db
```

In a second terminal, with `POSTGRES_HOST=localhost` and
`POSTGRES_PORT=15432` in your environment:

```bash
psql "postgresql://ragguard:<password>@localhost:15432/ragguard" -f db/init/01_extensions.sql -f db/init/02_schema.sql
```

```bash
uv run python scripts/fetch_corpus.py && uv run python scripts/seed.py && uv run python scripts/index_corpus.py
```

Indexing embeds 4,996 chunks locally and takes roughly seven minutes.

**6. Deploy.**

```bash
flyctl deploy --app ragguard
```

```bash
flyctl open --app ragguard
```

---

## Cost

The configuration in `fly.toml` stops the machine when idle
(`auto_stop_machines = "stop"`, `min_machines_running = 0`), which is right
for a demo that is idle almost all the time. The trade is a slow first
request after a quiet period while the machine starts.

Expect a few dollars a month, dominated by the Postgres volume rather than
the API. Fly has no free tier; a portfolio demo that nobody visits still
costs something.

---

## Verifying a deployment

```bash
curl https://ragguard.fly.dev/api/health
```

Expect `{"status":"ok","chunks":4996,...}`. A `chunks` count of 0 means step
5 did not complete — the API will start and return nothing for every query,
which is worse than failing outright.

Then check that identity actually gates results:

```bash
curl -s -X POST https://ragguard.fly.dev/api/token -H 'Content-Type: application/json' -d '{"persona":"newhire@gitlab.test"}'
```

Search with that token and again with `exec@gitlab.test`. The new hire
should see only `public` documents; the executive should see `restricted`
ones. If both return the same thing, the deployment is not enforcing
anything and should be taken down.

---

## When it breaks

**Health check fails, machine restarts in a loop.**
Almost always memory. Embedding a query runs an ONNX model in-process and
256MB is not enough — `fly.toml` asks for 1GB. Check with
`flyctl logs --app ragguard` and look for OOM kills.

**First request takes 30+ seconds, then everything is fast.**
Either the machine cold-started from stopped, which is expected, or the
embedding model is being downloaded at runtime. The Dockerfile pre-downloads
it; if that layer was skipped the symptom returns.

**Every search returns nothing, health is ok.**
The corpus was never indexed. `chunks` in the health response will be 0.

**Search works but "connected to" is always empty.**
Expected. Neo4j is not deployed, and the graph endpoint returns
`{"graph": "unavailable"}` rather than erroring. Deploy Neo4j separately and
set `NEO4J_*` if the graph view is wanted.

**Everyone sees the same results.**
Serious. Check `POSTGRES_*` point at the seeded database and that
`scripts/seed.py` ran — if the identity tables are empty every persona
resolves to the same baseline.

---

## Rollback

```bash
flyctl releases --app ragguard
```

```bash
flyctl deploy --image <previous image ref> --app ragguard
```

The database is unaffected by an application rollback. Schema changes are
not versioned — `db/init/` runs once on an empty volume — so a schema change
means recreating the database and re-running step 5.

---

## Building locally first

Always worth doing before a deploy, since it catches the same failures in
seconds rather than minutes:

```bash
docker build -t ragguard-api:local .
```

```bash
docker run --rm --network ragguard_default -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 -e POSTGRES_USER=ragguard -e POSTGRES_PASSWORD=ragguard_dev_pw -e POSTGRES_DB=ragguard -p 8100:8080 ragguard-api:local
```

Then `curl http://localhost:8100/api/health`.
