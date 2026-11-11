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

"""Bounded reconciliation and admission of Source-driven Family progress."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.supervision import ArtifactProcessingFence, ArtifactProcessingLeaseRepository
from powercontext.builtin.persistence.tables import PROFILE_POLICIES_TABLE, SOURCE_JOURNAL_HEADS_TABLE


async def enabled_profile_scopes(connection: AsyncConnection, scope_ids: tuple[str, ...]) -> frozenset[str]:
    """Qualify one bounded automatic page before accepting any invocation.

    Hold enabled Policy rows through admission so concurrent policy changes
    serialize with the decision. This reads metadata only, before any Worker
    authorization, model construction, or retry state exists for the page.
    """

    if not scope_ids:
        return frozenset()
    table = PROFILE_POLICIES_TABLE
    rows = await connection.scalars(
        select(table.c.scope_id)
        .where(table.c.scope_id.in_(scope_ids), table.c.generation_enabled.is_(True))
        .with_for_update()
    )
    return frozenset(rows)


class SourceProcessingPendingProvider:
    """Inspect journal metadata, never Source bodies, prompts, or models."""

    def __init__(self, database: AsyncDatabase, binding_name: str, artifact_family: str) -> None:
        self._database = database
        self._binding = binding_name
        self._family = artifact_family

    async def reconcile(self, after_scope_id: str | None, limit: int, *, fence: ArtifactProcessingFence) -> str | None:
        heads = SOURCE_JOURNAL_HEADS_TABLE
        statement = select(heads.c.scope_id, heads.c.position).order_by(heads.c.scope_id).limit(limit)
        if after_scope_id is not None:
            statement = statement.where(heads.c.scope_id > after_scope_id)
        if self._family == "profile":
            statement = statement.join(PROFILE_POLICIES_TABLE, PROFILE_POLICIES_TABLE.c.scope_id == heads.c.scope_id)
            statement = statement.where(PROFILE_POLICIES_TABLE.c.generation_enabled.is_(True))
        async with self._database.transaction() as connection:
            await ArtifactProcessingLeaseRepository().require_fence(connection, fence)
            rows = (await connection.execute(statement)).all()
            intents = ArtifactProcessingIntentRepository()
            cursors = SourceCursorRepository()
            for scope, high in rows:
                cursor = await cursors.load(connection, scope, self._binding)
                if int(high) <= (0 if cursor is None else cursor.cursor.sequence):
                    continue
                intent = await intents.load(connection, scope, self._binding, for_update=True)
                if intent is None or intent.dirty_generation == intent.clean_generation:
                    await intents.mark_dirty(connection, scope, self._binding)
        return str(rows[-1][0]) if len(rows) == limit else None


class SourceCountAdmission:
    """Accept one Scope whose pending Source count reached a configured threshold.

    The count is the number of persisted journal entries behind the family's last
    committed Source cursor. It follows the journal identity, so a replayed capture
    and a receipt trust upgrade keep their original position and never add to it. A
    lineage-only Source still occupies a journal entry and therefore counts, while the
    Family's own rules keep excluding it from generation evidence.
    """

    def __init__(self, database: AsyncDatabase, binding_name: str, threshold: int) -> None:
        if type(threshold) is not int or threshold < 1:
            raise ValueError("threshold must be a positive integer")  # noqa: TRY003
        self._database = database
        self._binding = binding_name
        self._threshold = threshold

    async def admit(
        self, after_scope_id: str | None, limit: int, *, fence: ArtifactProcessingFence
    ) -> tuple[str | None, bool]:
        """Accept every eligible Scope in one bounded page.

        Returns the next keyset position, or None when this pass is complete, and
        whether at least one invocation was accepted.
        """

        heads = SOURCE_JOURNAL_HEADS_TABLE
        statement = select(heads.c.scope_id, heads.c.position).order_by(heads.c.scope_id).limit(limit)
        if after_scope_id is not None:
            statement = statement.where(heads.c.scope_id > after_scope_id)
        async with self._database.transaction() as connection:
            await ArtifactProcessingLeaseRepository().require_fence(connection, fence)
            rows = (await connection.execute(statement)).all()
            if not rows:
                return None, False
            scope_ids = tuple(str(row[0]) for row in rows)
            cursors = await SourceCursorRepository().load_many(connection, scope_ids, self._binding)
            intents = ArtifactProcessingIntentRepository()
            accepted = False
            for scope, high in rows:
                scope_id = str(scope)
                cursor = cursors.get(scope_id)
                position = 0 if cursor is None else cursor.cursor.sequence
                if int(high) - position < self._threshold:
                    continue
                intent = await intents.load(connection, scope_id, self._binding, for_update=True)
                # An accepted but unhandled invocation already owns this Scope. The
                # threshold never discards it, and never accepts a second one.
                if intent is not None and intent.requested_generation > intent.handled_generation:
                    continue
                await intents.request(connection, scope_id, self._binding)
                accepted = True
        return (str(rows[-1][0]) if len(rows) == limit else None), accepted


__all__ = ["SourceCountAdmission", "SourceProcessingPendingProvider", "enabled_profile_scopes"]
