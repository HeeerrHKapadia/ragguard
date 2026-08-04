#!/usr/bin/env bash
# Start the bundled database, then the API.
#
# The data directory was built into the image, so there is nothing to
# initialise, migrate, or restore here — Postgres opens a cluster that is
# already populated. Startup is a few seconds rather than the fifteen minutes
# the build took.
set -euo pipefail

export PGDATA="${PGDATA:-/var/lib/ragguard/pgdata}"
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export POSTGRES_USER=ragguard
export POSTGRES_DB=ragguard
# The cluster uses trust auth on a loopback-only socket, so this is a
# placeholder the connection string needs rather than a credential.
export POSTGRES_PASSWORD=unused-trust-auth

# Spaces hands out an ephemeral filesystem, so the cluster starts fresh from
# the image on every boot. Any write made by a visitor is discarded when the
# container restarts, which for a public read-only demo is a feature.
pg_ctl -D "$PGDATA" -o "-c listen_addresses=127.0.0.1 -p 5432" -w start

# Stop the database on the way out rather than letting the container be
# killed with it running. Costs nothing and keeps logs clean.
trap 'pg_ctl -D "$PGDATA" -m fast -w stop || true' EXIT

# JWT_SECRET is a Space secret when one is set, and falls back to a random
# value per boot otherwise. A random per-boot secret invalidates tokens on
# restart, which for a demo is preferable to shipping a known key.
export JWT_SECRET="${JWT_SECRET:-$(head -c 32 /dev/urandom | base64)}"

exec uv run uvicorn ragguard.api.app:app --host 0.0.0.0 --port "${PORT:-7860}"
