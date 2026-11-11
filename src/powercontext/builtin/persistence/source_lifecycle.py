# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Durable registrations and Source event state for one Scope's lifecycle graph.

The graph itself is a directed acyclic dependency set whose nodes are registered
per Scope. Only the Source nodes own mutable state: their last applied event
sequence, the availability that event established, and the reason it carried.
Every derived node's availability is a pure function of its parents, so it is
recomputed from the Source state instead of being stored as a second truth.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE


@dataclass(frozen=True)
class SourceLifecycleNode:
    """One registered lifecycle node and the state it currently carries."""

    node_id: str
    kind: str
    ordinal: int
    reference: str
    parents: tuple[str, ...]
    policy: str | None
    revision: int | None
    sequence: int
    available: bool
    event_reason: str | None


class SourceLifecycleRepository:
    """Read and write one Scope's lifecycle registrations and Source event state."""

    async def nodes(
        self,
        connection: AsyncConnection,
        scope_id: str,
        /,
    ) -> tuple[SourceLifecycleNode, ...]:
        """Read every registered node of one Scope in registration order."""

        rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODES_TABLE)
                .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
                .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
            )
        ).mappings()
        return tuple(_node(row) for row in rows)

    async def register(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node: SourceLifecycleNode,
        /,
    ) -> None:
        """Insert one new node binding."""

        await connection.execute(
            insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                scope_id=scope_id,
                node_id=node.node_id,
                kind=node.kind,
                ordinal=node.ordinal,
                reference=node.reference.encode("utf-8"),
                parents=_encode_parents(node.parents),
                policy=node.policy,
                revision=node.revision,
                sequence=node.sequence,
                available=node.available,
                event_reason=node.event_reason,
            )
        )

    async def record_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        *,
        sequence: int,
        available: bool,
        reason: str,
        states: Mapping[str, bool],
    ) -> None:
        """Record one applied Source event together with the derived node states."""

        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(sequence=sequence, available=available, event_reason=reason)
        )
        for target, state in states.items():
            await connection.execute(
                update(SOURCE_LIFECYCLE_NODES_TABLE)
                .where(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == target,
                )
                .values(available=state)
            )


class SourceLifecycleStore:
    """Open one transaction per lifecycle read or write."""

    def __init__(self, database: AsyncDatabase) -> None:
        self._database = database
        self._repository = SourceLifecycleRepository()

    async def nodes(self, scope_id: str, /) -> tuple[SourceLifecycleNode, ...]:
        async with self._database.transaction() as connection:
            return await self._repository.nodes(connection, scope_id)

    async def register(self, scope_id: str, node: SourceLifecycleNode, /) -> None:
        async with self._database.transaction() as connection:
            await self._repository.register(connection, scope_id, node)

    async def record_event(
        self,
        scope_id: str,
        node_id: str,
        *,
        sequence: int,
        available: bool,
        reason: str,
        states: Mapping[str, bool],
    ) -> None:
        async with self._database.transaction() as connection:
            await self._repository.record_event(
                connection,
                scope_id,
                node_id,
                sequence=sequence,
                available=available,
                reason=reason,
                states=states,
            )


def _encode_parents(parents: tuple[str, ...]) -> bytes:
    return json.dumps(list(parents), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _node(row: Mapping[str, object]) -> SourceLifecycleNode:
    parents = row["parents"]
    return SourceLifecycleNode(
        node_id=str(row["node_id"]),
        kind=str(row["kind"]),
        ordinal=int(row["ordinal"]),  # type: ignore[arg-type]
        reference=bytes(row["reference"]).decode("utf-8"),  # type: ignore[arg-type]
        parents=tuple(json.loads(bytes(parents).decode("utf-8"))),  # type: ignore[arg-type]
        policy=None if row["policy"] is None else str(row["policy"]),
        revision=None if row["revision"] is None else int(row["revision"]),  # type: ignore[arg-type]
        sequence=int(row["sequence"]),  # type: ignore[arg-type]
        available=bool(row["available"]),
        event_reason=None if row["event_reason"] is None else str(row["event_reason"]),
    )


__all__ = [
    "SourceLifecycleNode",
    "SourceLifecycleRepository",
    "SourceLifecycleStore",
]
