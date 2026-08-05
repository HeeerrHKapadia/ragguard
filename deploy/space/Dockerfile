# Single-container image for Hugging Face Spaces.
#
# Spaces gives one container, no managed database, and no persistent disk on
# the free tier. So Postgres runs inside this image and the corpus is seeded
# and embedded at BUILD time, baked into the image layers.
#
# That inverts the usual trade deliberately. The image is large and the build
# is slow — roughly fifteen minutes, most of it embedding 4,996 chunks — but
# the container starts with data already present, needs no volume, and costs
# nothing to run. For a demo that is idle most of the time and must be free,
# that is the right way round.
#
# Spaces constraints this has to satisfy:
#   - listen on 7860
#   - run as UID 1000, which cannot write to the default Postgres data dir
#   - no volumes, no external services

FROM pgvector/pgvector:pg17

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PGDATA=/var/lib/ragguard/pgdata \
    FASTEMBED_CACHE_PATH=/opt/models \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Spaces runs as UID 1000. Everything this image writes at build or run time
# has to belong to that user, including the Postgres data directory — the
# packaged `postgres` user is not who will be executing here.
RUN useradd -m -u 1000 -s /bin/bash app \
    && mkdir -p /var/lib/ragguard /opt/models /opt/python /app \
    && chown -R app:app /var/lib/ragguard /opt/models /opt/python /app

USER app
WORKDIR /app

RUN uv python install 3.13

COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app src/ ./src/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app db/ ./db/
RUN uv sync --frozen --no-dev

# Cache the embedding model in the image. Downloading 133MB on first request
# looks exactly like a hung deployment.
RUN uv run python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
print('model cached')"

# --- build the database into the image -------------------------------------
#
# One RUN layer: initialise the cluster, start it, load everything, stop it
# cleanly. Splitting this across layers would lose the running server between
# them, and a Postgres shut down by layer boundary rather than by `pg_ctl
# stop` leaves a data directory that needs recovery on first boot.
ENV POSTGRES_HOST=127.0.0.1 \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=ragguard \
    POSTGRES_DB=ragguard \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    EMBEDDING_DIM=384 \
    # Half the corpus. Embedding dominates the build and a free Space build
    # has a time limit, so the demo runs on 300 documents where the measured
    # results use 639.
    #
    # Restricted documents are never sampled away, so all 52 survive the cut
    # and the executive still sees material an engineer cannot. A smaller
    # corpus that lost the sensitive tier would demonstrate the opposite of
    # the point. 100 rather than something lower because below that GitLab
    # drops to two or three public documents, and a new hire receiving almost
    # nothing reads as a broken demo rather than a correctly restricted one.
    MAX_DOCS_PER_TENANT=100

# POSTGRES_PASSWORD is set per-RUN rather than as an ENV layer. The cluster
# is created with trust auth and listens only on 127.0.0.1 inside the
# container, so the value is never checked — but baking a variable named
# like a credential into the image metadata is the kind of thing that gets
# copied into a project where it does matter.
RUN set -eux; \
    export POSTGRES_PASSWORD=unused-trust-auth; \
    initdb -D "$PGDATA" -U ragguard --auth=trust >/dev/null; \
    pg_ctl -D "$PGDATA" -o "-c listen_addresses=127.0.0.1 -p 5432" -w start; \
    createdb -h 127.0.0.1 -U ragguard ragguard; \
    psql -h 127.0.0.1 -U ragguard -d ragguard -q -f db/init/01_extensions.sql; \
    psql -h 127.0.0.1 -U ragguard -d ragguard -q -f db/init/02_schema.sql; \
    uv run python scripts/fetch_corpus.py; \
    uv run python scripts/seed.py; \
    uv run python scripts/index_corpus.py; \
    rm -rf data/raw; \
    psql -h 127.0.0.1 -U ragguard -d ragguard -c "VACUUM ANALYZE"; \
    pg_ctl -D "$PGDATA" -w stop

COPY --chown=app:app deploy/space/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 7860
CMD ["/app/entrypoint.sh"]
