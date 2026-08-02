"""Configuration loaded from environment variables.

One place that reads the environment, so nothing else in the codebase calls
os.getenv directly. When Phase 4 adds an auth service and Phase 6 adds
tracing, their settings land here too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = three levels up from this file (src/ragguard/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Loads .env into the process environment. Real environment variables win,
# which is what you want in CI and production.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    pg_user: str
    pg_password: str
    pg_db: str
    pg_host: str
    pg_port: int
    embedding_model: str
    embedding_dim: int
    connect_timeout: int = 5

    @property
    def dsn(self) -> str:
        """Postgres connection string.

        connect_timeout matters more than it looks: without it, a connection
        to a host where nothing is listening can hang for the OS-level TCP
        timeout instead of failing fast with a useful message.
        """
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
            f"?connect_timeout={self.connect_timeout}"
        )

    @property
    def safe_dsn(self) -> str:
        """Same DSN with the password masked — safe to log or print."""
        return (
            f"postgresql://{self.pg_user}:***"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


def load_settings() -> Settings:
    return Settings(
        pg_user=os.getenv("POSTGRES_USER", "ragguard"),
        pg_password=os.getenv("POSTGRES_PASSWORD", "ragguard_dev_pw"),
        pg_db=os.getenv("POSTGRES_DB", "ragguard"),
        pg_host=os.getenv("POSTGRES_HOST", "localhost"),
        pg_port=int(os.getenv("POSTGRES_PORT", "5433")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
    )


settings = load_settings()
