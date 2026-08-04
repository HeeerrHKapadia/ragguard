# Production image for the API.
#
# Two things are deliberate.
#
# The embedding model is downloaded at build time rather than on first
# request. It is 133MB, and a cold start that quietly stalls for thirty
# seconds while fetching weights looks exactly like a broken deployment.
#
# Dependencies install before the source is copied, so editing application
# code does not invalidate the dependency layer. That is the difference
# between a ten second rebuild and a three minute one.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/opt/models

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application layer.
COPY src/ ./src/
COPY static/ ./static/
COPY config/ ./config/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev

# Pre-download the embedding model so the first request does not pay for it.
RUN uv run python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
print('embedding model cached')"

EXPOSE 8080

# One worker: the process holds a single database connection and an in-memory
# trace buffer, neither of which is shared across workers. Scaling out means
# more machines, not more workers inside one.
CMD ["uv", "run", "uvicorn", "ragguard.api.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
