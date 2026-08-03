"""Exercise the API end to end, including the parts a unit test cannot reach.

Runs the real application through FastAPI's test transport, so startup,
dependency wiring, the database, and the graph are all live. The unit tests
in tests/test_api_auth.py prove tokens cannot be forged; this proves the
service actually refuses to serve documents to the wrong person.

The central assertion is the demo in miniature: the same query, asked by
three personas, must return progressively more — and a persona must never
receive anything above their clearance.

Run:  uv run python scripts/api_smoke.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from ragguard.access import TIER_RANK
from ragguard.api.app import app

QUERY = "compensation review and pay bands"

# Least to most privileged, within one tenant.
LADDER = ["newhire@gitlab.test", "eng@gitlab.test", "exec@gitlab.test"]

MAX_TIER = {
    "newhire@gitlab.test": "public",
    "eng@gitlab.test": "internal",
    "exec@gitlab.test": "restricted",
}

PASS = "  [ok]  "
FAIL = "  [FAIL]"


def main() -> int:
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"{PASS if ok else FAIL} {label}{f' — {detail}' if detail else ''}")
        if not ok:
            failures.append(label)

    with TestClient(app) as client:
        health = client.get("/api/health")
        check(health.status_code == 200, "health", f"{health.json().get('chunks')} chunks")

        check(
            client.post("/api/search", json={"query": QUERY}).status_code == 401,
            "unauthenticated search refused",
        )
        check(
            client.post(
                "/api/search", json={"query": QUERY},
                headers={"Authorization": "Bearer forged.token.value"},
            ).status_code == 401,
            "forged token refused",
        )

        print()
        results = {}
        for persona in LADDER:
            token = client.post("/api/token", json={"persona": persona}).json()["token"]
            response = client.post(
                "/api/search", json={"query": QUERY, "k": 8},
                headers={"Authorization": f"Bearer {token}"},
            )
            check(response.status_code == 200, f"search as {persona}")
            body = response.json()
            results[persona] = body

            tiers = {r["tier"] for r in body["results"]}
            ceiling = TIER_RANK[MAX_TIER[persona]]
            over = {t for t in tiers if TIER_RANK[t] > ceiling}
            check(not over, f"{persona} stays within clearance",
                  f"tiers {sorted(tiers)}" if not over else f"LEAKED {sorted(over)}")

            check(
                bool(body["trace"]["spans"]),
                f"{persona} request traced",
                f"{body['trace']['total_ms']} ms",
            )

        print()
        # Privilege must actually change the answer, or the demo is a lie.
        newhire_uris = {r["uri"] for r in results[LADDER[0]]["results"]}
        exec_uris = {r["uri"] for r in results[LADDER[-1]]["results"]}
        check(newhire_uris != exec_uris, "privilege changes the results",
              f"{len(newhire_uris & exec_uris)} of {len(exec_uris)} shared")

        # The graph endpoint must respect the same boundary.
        token = client.post(
            "/api/token", json={"persona": "newhire@gitlab.test"}
        ).json()["token"]
        if newhire_uris:
            neighbours = client.get(
                "/api/graph", params={"uri": next(iter(newhire_uris))},
                headers={"Authorization": f"Bearer {token}"},
            ).json()["neighbours"]
            over = [n for n in neighbours if TIER_RANK[n["tier"]] > TIER_RANK["public"]]
            check(not over, "graph neighbours stay within clearance",
                  f"{len(neighbours)} reachable")

        check(
            client.get("/api/graph", params={"uri": "x"}).status_code == 401,
            "unauthenticated graph refused",
        )

    print()
    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}\n")
        return 1
    print("API behaves correctly for every persona.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
