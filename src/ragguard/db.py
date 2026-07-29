"""Database connection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from ragguard.config import settings


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a connection with the pgvector type adapter registered.

    Without register_vector(), psycopg hands back embedding columns as raw
    strings instead of numpy arrays, and refuses to accept numpy arrays as
    query parameters. Easy to forget; confusing to debug.
    """
    with psycopg.connect(settings.dsn, autocommit=autocommit) as conn:
        register_vector(conn)
        yield conn
