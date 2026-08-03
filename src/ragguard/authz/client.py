"""Thin OpenFGA client.

Wraps store creation, model upload, tuple writes and the two read paths that
matter for retrieval — checking one object, and listing everything a user can
reach. Which of those two a system uses decides its whole architecture, which
is what scripts/authz_benchmark.py measures.
"""

from __future__ import annotations

import os
from typing import Self

from openfga_sdk import ClientConfiguration, OpenFgaClient
from openfga_sdk.client.models import (
    ClientCheckRequest,
    ClientListObjectsRequest,
    ClientTuple,
    ClientWriteRequest,
)
from openfga_sdk.models import CreateStoreRequest, WriteAuthorizationModelRequest

from ragguard.authz.model import Tuple, build_model

# OpenFGA rejects oversized write batches; 100 is the documented default cap.
WRITE_BATCH = 100


def api_url() -> str:
    port = os.getenv("OPENFGA_PORT", "8090")
    return f"http://localhost:{port}"


class Authz:
    """Owns a store, its model, and its tuples."""

    def __init__(self, store_id: str | None = None, model_id: str | None = None) -> None:
        self.store_id = store_id
        self.model_id = model_id

    async def __aenter__(self) -> Self:
        config = ClientConfiguration(api_url=api_url(), store_id=self.store_id)
        self._client = OpenFgaClient(config)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.close()

    async def create_store(self, name: str = "ragguard") -> str:
        response = await self._client.create_store(CreateStoreRequest(name=name))
        self.store_id = response.id
        self._client.set_store_id(self.store_id)
        return self.store_id

    async def write_model(self) -> str:
        """Upload the authorization model."""
        response = await self._client.write_authorization_model(
            WriteAuthorizationModelRequest(
                schema_version="1.1",
                type_definitions=build_model(),
            )
        )
        self.model_id = response.authorization_model_id
        return self.model_id

    async def write_tuples(self, tuples: list[Tuple]) -> int:
        written = 0
        for start in range(0, len(tuples), WRITE_BATCH):
            batch = tuples[start:start + WRITE_BATCH]
            await self._client.write(
                ClientWriteRequest(
                    writes=[
                        ClientTuple(user=t.user, relation=t.relation, object=t.object)
                        for t in batch
                    ]
                )
            )
            written += len(batch)
        return written

    async def delete_tuples(self, tuples: list[Tuple]) -> None:
        for start in range(0, len(tuples), WRITE_BATCH):
            batch = tuples[start:start + WRITE_BATCH]
            await self._client.write(
                ClientWriteRequest(
                    deletes=[
                        ClientTuple(user=t.user, relation=t.relation, object=t.object)
                        for t in batch
                    ]
                )
            )

    async def check(self, user: str, relation: str, obj: str) -> bool:
        """One question about one object. Authoritative, and one round trip."""
        response = await self._client.check(
            ClientCheckRequest(user=user, relation=relation, object=obj)
        )
        return bool(response.allowed)

    async def list_objects(self, user: str, relation: str, obj_type: str) -> list[str]:
        """Everything of a type this user can reach.

        The call that decides whether an external authorization service can
        be used as a pre-filter or only as a post-filter — and Phase 2
        already measured what post-filtering costs low-privilege users.
        """
        response = await self._client.list_objects(
            ClientListObjectsRequest(user=user, relation=relation, type=obj_type)
        )
        return list(response.objects)
