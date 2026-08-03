"""Bearer tokens, and where identity comes from.

The identity in the token is the *only* thing that decides what a request
can see. Nothing downstream accepts a persona, tenant, or clearance from the
request body — if it did, changing what you can read would be a matter of
editing a JSON field, which is the whole class of bug this project exists to
avoid.

**This is demo authentication.** `issue_token` hands out a token for any
seeded persona with no password, because the point of the demo is switching
between personas to watch results change. A real deployment replaces this
endpoint with an identity provider and keeps everything below it unchanged —
the token is verified the same way either way.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ALGORITHM = "HS256"
TOKEN_TTL = timedelta(hours=8)

bearer = HTTPBearer(auto_error=False)


def secret() -> str:
    # Dev default is fine here because the tokens grant access to three
    # public company handbooks. A deployment holding anything real must set
    # this, and a startup check would be the next thing to add.
    return os.getenv("JWT_SECRET", "ragguard-dev-secret-not-for-production")


def issue_token(email: str, tenant: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": email, "tenant": tenant, "iat": now, "exp": now + TOKEN_TTL},
        secret(),
        algorithm=ALGORITHM,
    )


def current_identity(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> str:
    """Return the caller's persona id, or refuse the request.

    Every failure path returns the same message. Distinguishing "no token"
    from "expired" from "bad signature" is a small courtesy to a legitimate
    user and a useful oracle to everyone else.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to search.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(credentials.credentials, secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to search.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in to search.")
    return subject
