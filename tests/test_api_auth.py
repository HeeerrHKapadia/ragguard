"""Tests for the token layer.

No database or running server needed — the token is self-contained, and the
question these ask is whether identity can be forged or smuggled in, which
is decided before anything is retrieved.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from ragguard.api.auth import ALGORITHM, current_identity, issue_token, secret

# Long enough that PyJWT does not warn about HMAC key length — the warning
# is correct advice, and a test fixture is no reason to make output noisy.
WRONG_SECRET = "a-different-secret-that-is-long-enough-for-sha256"


def creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class TestIdentityResolution:
    def test_valid_token_resolves(self):
        token = issue_token("eng@acme.test", "acme")
        assert current_identity(creds(token)) == "eng@acme.test"

    def test_missing_token_is_refused(self):
        with pytest.raises(HTTPException) as exc:
            current_identity(None)
        assert exc.value.status_code == 401


class TestForgery:
    def test_wrong_signature_is_refused(self):
        forged = jwt.encode(
            {"sub": "exec@acme.test", "tenant": "acme"},
            WRONG_SECRET,
            algorithm=ALGORITHM,
        )
        with pytest.raises(HTTPException):
            current_identity(creds(forged))

    def test_unsigned_token_is_refused(self):
        # The classic JWT attack: claim the algorithm is "none" and drop the
        # signature. Rejected because decode is pinned to one algorithm.
        unsigned = jwt.encode({"sub": "exec@acme.test"}, key="", algorithm="none")
        with pytest.raises(HTTPException):
            current_identity(creds(unsigned))

    def test_expired_token_is_refused(self):
        past = datetime.now(UTC) - timedelta(hours=1)
        expired = jwt.encode(
            {"sub": "eng@acme.test", "exp": past}, secret(), algorithm=ALGORITHM
        )
        with pytest.raises(HTTPException):
            current_identity(creds(expired))

    def test_token_without_subject_is_refused(self):
        anonymous = jwt.encode({"tenant": "acme"}, secret(), algorithm=ALGORITHM)
        with pytest.raises(HTTPException):
            current_identity(creds(anonymous))

    def test_garbage_is_refused(self):
        with pytest.raises(HTTPException):
            current_identity(creds("not-a-token-at-all"))


class TestFailuresAreIndistinguishable:
    def test_every_rejection_says_the_same_thing(self):
        """A distinct message per failure is an oracle for an attacker.

        Knowing whether a token was expired, forged, or simply absent tells
        someone probing the API which part of their guess was right.
        """
        bad = [
            None,
            creds("garbage"),
            creds(jwt.encode({"sub": "x"}, WRONG_SECRET, algorithm=ALGORITHM)),
            creds(jwt.encode(
                {"sub": "x", "exp": datetime.now(UTC) - timedelta(hours=1)},
                secret(), algorithm=ALGORITHM,
            )),
        ]
        details = set()
        for candidate in bad:
            with pytest.raises(HTTPException) as exc:
                current_identity(candidate)
            details.add(exc.value.detail)
        assert len(details) == 1


class TestTokenContents:
    def test_token_carries_tenant_and_expiry(self):
        token = issue_token("eng@acme.test", "acme")
        payload = jwt.decode(token, secret(), algorithms=[ALGORITHM])
        assert payload["tenant"] == "acme"
        assert payload["exp"] > time.time()
