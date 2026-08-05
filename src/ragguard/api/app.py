"""The service.

Endpoints are deliberately few. The interesting part is not the API surface
but that every one of them resolves identity from the token and nothing
else, and that the search path is the same pre-filtered retriever the
evaluation measured — not a reimplementation that could drift from the
numbers in the README.

Run:  uv run uvicorn ragguard.api.app:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragguard.access import TIER_RANK, Principal, load_principals
from ragguard.api.auth import current_identity, issue_token
from ragguard.api.tracing import recent, span, trace
from ragguard.config import PROJECT_ROOT, settings
from ragguard.db import connect
from ragguard.eval.dataset import GoldenCase
from ragguard.graph.filters import visibility_cypher, visibility_params
from ragguard.graph.store import graph_driver
from ragguard.retrieval.dense import PreFilterRetriever

STATIC = PROJECT_ROOT / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Hold one connection and one driver for the process lifetime.

    Reconnecting per request would dominate the latency the traces are
    supposed to be measuring.

    Both context managers are kept in variables that outlive this block.
    `connect().__enter__()` looks equivalent and is not: the generator
    context manager it returns would have no remaining reference, get
    collected, run its own cleanup, and close the connection out from under
    the server. Second time this exact shape has caused a bug in this
    project, hence the comment.
    """
    conn_ctx = connect()
    conn = conn_ctx.__enter__()

    # The graph is optional. Search — the thing the demo is actually about —
    # needs only Postgres, so a deployment can run two services instead of
    # three and the "connected to" expansion degrades to empty rather than
    # taking the whole service down with it.
    driver_ctx = None
    driver = None
    try:
        driver_ctx = graph_driver()
        driver = driver_ctx.__enter__()
        driver.verify_connectivity()
    except Exception:  # noqa: BLE001 - any failure means the same thing
        if driver_ctx is not None:
            driver_ctx.__exit__(None, None, None)
        driver_ctx = None
        driver = None

    cur = conn.cursor()
    state["conn"] = conn
    state["driver"] = driver
    state["principals"] = load_principals(cur)
    state["retriever"] = PreFilterRetriever(conn)

    cur.execute(
        """SELECT u.email, u.display_name, u.title, t.slug
             FROM users u JOIN tenants t ON t.id = u.tenant_id
         ORDER BY t.slug, u.email"""
    )
    # Ordered by privilege rather than alphabetically, so the dropdown reads
    # bottom to top and stepping through it tells the story: results appear
    # as clearance rises. Alphabetical put the new hire last, which buries
    # the comparison the demo exists to make.
    directory = [
        {"email": e, "name": n, "title": ti, "tenant": s} for e, n, ti, s in cur.fetchall()
    ]
    principals = state["principals"]
    state["directory"] = sorted(
        directory,
        key=lambda p: (
            p["tenant"],
            TIER_RANK[principals[p["email"]].max_clearance]
            if p["email"] in principals else 0,
            p["email"],
        ),
    )

    try:
        yield
    finally:
        if driver_ctx is not None:
            driver_ctx.__exit__(None, None, None)
        conn_ctx.__exit__(None, None, None)


app = FastAPI(title="ragguard", lifespan=lifespan)


def principal_for(email: str) -> Principal:
    principal = state["principals"].get(email)
    if principal is None:
        raise HTTPException(status_code=401, detail="Sign in to search.")
    return principal


class TokenRequest(BaseModel):
    persona: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=8, ge=1, le=25)


@app.get("/api/personas")
def personas() -> list[dict]:
    """Who can be signed in as. Demo scaffolding, not a real directory."""
    return state["directory"]


@app.post("/api/token")
def token(request: TokenRequest) -> dict:
    principal = principal_for(request.persona)
    return {"token": issue_token(request.persona, principal.tenant_slug)}


@app.post("/api/search")
def search(request: SearchRequest, email: str = Depends(current_identity)) -> dict:
    principal = principal_for(email)

    with trace(email, request.query) as record:
        case = GoldenCase(
            case_id="api", tenant=principal.tenant_slug, persona=email,
            query=request.query, query_class="local", relevant_uris=(),
        )
        with span(record, "retrieve"):
            found = state["retriever"].retrieve(case, principal, request.k)

        with span(record, "hydrate"):
            titles = {}
            if found:
                cur = state["conn"].cursor()
                cur.execute(
                    "SELECT source_uri, title FROM documents WHERE source_uri = ANY(%s)",
                    ([d.uri for d in found],),
                )
                titles = dict(cur.fetchall())

        record.results = len(found)

    return {
        "persona": email,
        "results": [
            {
                "uri": d.uri,
                "title": titles.get(d.uri, d.uri),
                "section": d.section,
                "tier": d.tier,
                # Cosine distance inverted so larger reads as better in a UI.
                "relevance": round(1 - d.score, 3),
            }
            for d in found
        ],
        "trace": record.as_dict(),
    }


@app.get("/api/graph")
def graph(uri: str, email: str = Depends(current_identity)) -> dict:
    """Neighbours of a document that this persona may traverse.

    Returns only what the caller can reach. It does not report how many
    neighbours were withheld — that count is the existence-disclosure risk
    recorded in Phase 8, and an endpoint that served it would turn a latent
    weakness into a supported feature.
    """
    principal = principal_for(email)
    if state["driver"] is None:
        return {"origin": uri, "neighbours": [], "graph": "unavailable"}

    params = visibility_params(principal)
    params["uri"] = uri

    with state["driver"].session() as session:
        rows = session.run(
            f"""MATCH (s:Document {{uri: $uri}})
                WHERE {visibility_cypher('s')}
                MATCH p = (s)-[:LINKS_TO|MENTIONS*1..2]-(n:Document)
                WHERE n.uri <> s.uri
                  AND all(node IN nodes(p) WHERE
                        CASE
                          WHEN node:Concept  THEN
                            exists {{ MATCH (node)<-[:MENTIONS]-(w:Document)
                                      WHERE {visibility_cypher('w')} }}
                          WHEN node:Document THEN {visibility_cypher('node')}
                          ELSE false
                        END)
                RETURN DISTINCT n.uri AS uri, n.title AS title,
                       n.section AS section, n.tier AS tier
                LIMIT 25""",
            **params,
        ).data()

    return {"origin": uri, "neighbours": rows}


@app.get("/api/traces")
def traces() -> list[dict]:
    return recent()


@app.get("/api/health")
def health() -> dict:
    try:
        cur = state["conn"].cursor()
        cur.execute("SELECT count(*) FROM chunks")
        chunks = cur.fetchone()[0]
    except psycopg.Error:
        raise HTTPException(status_code=503, detail="Search is unavailable.") from None
    return {"status": "ok", "chunks": chunks, "model": settings.embedding_model}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
