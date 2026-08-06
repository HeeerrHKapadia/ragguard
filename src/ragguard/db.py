"""Database connection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ragguard.config import settings

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Register pgvector so embeddings round-trip as arrays, not strings."""
    register_vector(conn)


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a one-off connection with the pgvector type adapter registered.

    Prefer `get_pool()` for the API and any code that issues concurrent
    queries — a single shared connection serializes every request.
    """
    with psycopg.connect(settings.dsn, autocommit=autocommit) as conn:
        _configure(conn)
        yield conn


def get_pool(min_size: int = 2, max_size: int = 8) -> ConnectionPool:
    """Process-wide pool. Created once; closed explicitly on shutdown."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": False},
            configure=_configure,
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
