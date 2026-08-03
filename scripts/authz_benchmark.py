"""Verify OpenFGA against the oracle, then measure what it costs to use.

Two things matter here and only one of them is correctness.

**Parity.** This is the fourth implementation of one policy — Python oracle,
SQL, Cypher, and now a relationship graph. It has to admit exactly what the
oracle admits, or the answer a user gets depends on which code path ran.

**Architecture.** An external authorization service offers two read paths,
and choosing between them decides the shape of the whole system:

    Check         one object, one round trip. Authoritative and exact, but
                  filtering 300 documents means 300 calls.
    ListObjects   everything a user can reach, in one call. Usable as a
                  pre-filter — if it is fast enough and complete.

Phase 2 measured what post-filtering costs: the least privileged persona
lost 14.3 points of recall, and closing that gap needed 50x oversampling.
If Check is the only affordable path, an authorization service pushes the
architecture straight back into the pattern that was already measured as
worse. So this measures whether ListObjects is viable, rather than assuming
the vendor's recommended pattern is the right one.

Run:  uv run python scripts/authz_benchmark.py
"""

from __future__ import annotations

import asyncio
import pathlib
import statistics
import sys
import time

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.access import can_read, load_principals
from ragguard.authz.client import Authz
from ragguard.authz.model import Tuple, scoped
from ragguard.config import PROJECT_ROOT
from ragguard.db import connect

STORE_FILE = PROJECT_ROOT / ".authz_store"
CHECK_SAMPLE = 40


def doc_object(tenant: str, uri: str) -> str:
    return scoped("document", tenant, uri)


async def main() -> int:
    if not STORE_FILE.exists():
        print("Load the tuples first: uv run python scripts/load_authz.py")
        return 1
    store_id, model_id = STORE_FILE.read_text(encoding="utf-8").split()

    try:
        with connect() as conn:
            cur = conn.cursor()
            principals = load_principals(cur)
            cur.execute(
                """SELECT t.slug, d.source_uri, d.section, d.sensitivity
                     FROM documents d JOIN tenants t ON t.id = d.tenant_id"""
            )
            documents = cur.fetchall()
    except psycopg.OperationalError as exc:
        print(f"Could not reach Postgres: {str(exc).strip()}")
        return 1

    async with Authz(store_id=store_id, model_id=model_id) as authz:
        # --- parity -------------------------------------------------------
        print(f"\n{len(principals)} personas x {len(documents)} documents\n")
        print(f"  {'persona':<28} {'oracle':>7} {'openfga':>8} {'status':>10} {'ms':>7}")
        print("  " + "-" * 64)

        mismatches = 0
        list_times: list[float] = []

        for email in sorted(principals):
            principal = principals[email]
            user_obj = scoped("user", principal.tenant_slug, email)

            oracle = {
                doc_object(tenant, uri)
                for tenant, uri, section, tier in documents
                if can_read(principal, tenant, section, tier)
            }

            started = time.perf_counter()
            allowed = set(await authz.list_objects(user_obj, "viewer", "document"))
            elapsed = (time.perf_counter() - started) * 1000
            list_times.append(elapsed)

            extra = allowed - oracle
            missing = oracle - allowed
            status = "ok" if not (extra or missing) else "MISMATCH"
            mismatches += len(extra) + len(missing)

            print(f"  {email:<28} {len(oracle):>7} {len(allowed):>8} "
                  f"{status:>10} {elapsed:>6.0f}")
            for obj in sorted(extra)[:2]:
                print(f"      openfga admits, oracle forbids : {obj}")
            for obj in sorted(missing)[:2]:
                print(f"      openfga blocks, oracle allows  : {obj}")

        if mismatches:
            print(f"\n  POLICY MISMATCH: {mismatches} disagreements. The oracle is right.\n")
            return 1

        print("\n  OpenFGA is identical to the reference implementation.")

        # --- the two read paths -------------------------------------------
        probe = min(principals)
        probe_principal = principals[probe]
        probe_user = scoped("user", probe_principal.tenant_slug, probe)
        probe_docs = [
            doc_object(tenant, uri)
            for tenant, uri, _s, _t in documents
            if tenant == probe_principal.tenant_slug
        ][:CHECK_SAMPLE]

        check_times: list[float] = []
        for obj in probe_docs:
            started = time.perf_counter()
            await authz.check(probe_user, "viewer", obj)
            check_times.append((time.perf_counter() - started) * 1000)

        tenant_size = sum(
            1 for tenant, _u, _s, _t in documents if tenant == probe_principal.tenant_slug
        )
        per_check = statistics.median(check_times)

        print("\n\n  Read paths\n")
        print(f"    Check, one object          {per_check:>7.1f} ms")
        print(f"    Check, {tenant_size} documents      "
              f"{per_check * tenant_size:>7.0f} ms  (extrapolated)")
        print(f"    ListObjects, all at once   {statistics.median(list_times):>7.1f} ms")

        speedup = (per_check * tenant_size) / statistics.median(list_times)
        print(f"\n    ListObjects is {speedup:.0f}x faster than checking each document.")

        # --- revocation ----------------------------------------------------
        # A permission system is only as good as how fast it forgets. This
        # removes a group membership and measures when the decision flips.
        victim = next(
            (e for e in sorted(principals) if e.startswith("eng@")), None
        )
        if victim:
            v_principal = principals[victim]
            v_user = scoped("user", v_principal.tenant_slug, victim)
            before = len(await authz.list_objects(v_user, "viewer", "document"))

            membership = Tuple(
                user=v_user,
                relation="member",
                object=scoped("group", v_principal.tenant_slug, "engineering"),
            )

            started = time.perf_counter()
            await authz.delete_tuples([membership])
            after = len(await authz.list_objects(v_user, "viewer", "document"))
            revoke_ms = (time.perf_counter() - started) * 1000

            await authz.write_tuples([membership])
            restored = len(await authz.list_objects(v_user, "viewer", "document"))

            print("\n\n  Revocation\n")
            print(f"    {victim} before      {before:>4} documents")
            print(f"    after removing group membership   {after:>4} documents")
            print(f"    after restoring it                {restored:>4} documents")
            print(f"    propagation                    {revoke_ms:>7.1f} ms")
            print(
                "\n    Immediate, because access is computed from tuples at read\n"
                "    time rather than materialised into an index that would have\n"
                "    to be rebuilt. The stale-permission window is the round trip."
            )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
