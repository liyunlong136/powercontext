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

"""Source lifecycle propagation over registered evidence dependencies.

A Scope registers an evidence graph: Source nodes own the original evidence,
Memory nodes and Topic Memory nodes derive from them. A Source withdrawal event
propagates along the graph under each node's ``all`` / ``any`` policy, so the
nodes that lost their support stop being searchable while the nodes that kept
it stay available. Withdrawal keeps the body: an inactive Memory entry is still
readable through its exact citation, and a restored Source makes the nodes it
supports available again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from powercontext._logging import log_safely
from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import MemoryService
from powercontext.builtin.artifacts.topic_memory import TopicMemory, TopicMemorySearchResult
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import SourceLifecycleNode, SourceLifecycleStore
from powercontext.builtin.records import BaseValueNotFoundError, InvalidBaseAccessRequestError
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

logger = logging.getLogger(__name__)

SOURCE_NODE: Final[str] = "source"
MEMORY_NODE: Final[str] = "memory"
TOPIC_NODE: Final[str] = "topic"

ALL_POLICY: Final[str] = "all"
ANY_POLICY: Final[str] = "any"

GENERATION_PURPOSE: Final[str] = "generation"
AUDIT_PURPOSE: Final[str] = "audit"

# The Memory Artifact keeps a bounded audit reason; a longer event reason is
# truncated instead of failing the whole propagation.
_MAX_MEMORY_REASON_LENGTH: Final[int] = 512

LifecycleNodeKind = Literal["source", "memory", "topic"]
LifecyclePolicy = Literal["all", "any"]
LifecyclePurpose = Literal["generation", "audit"]


class SourceLifecycleError(ValueError):
    """Report a lifecycle registration, event, or resolution that breaks the contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        messages = {
            "conflict": "the node identifier already names a different lifecycle binding",
            "cycle": "the requested parents would close a dependency cycle",
            "duplicate-parent": "a parent node may not be listed more than once",
            "event-conflict": "the event sequence was already used by different content",
            "event-revision": "the event revision does not match the Source lifecycle revision",
            "event-sequence": "the event sequence does not advance the Source event log",
            "invalid-event": "a lifecycle event must be a validated SourceLifecycleEvent",
            "invalid-family": "a Topic node requires a Topic Memory Artifact reference",
            "invalid-node-id": "the node identifier must be a non-empty string",
            "invalid-policy": "the dependency policy must be all or any",
            "invalid-revision": "the lifecycle revision must be a positive integer",
            "missing-parent": "a parent node is not registered in this Scope",
            "missing-reference": "the registered reference does not exist or cannot be read",
            "no-parents": "a derived node requires at least one parent",
            "not-a-source": "a lifecycle event may only target a registered Source",
            "not-registered": "the node is not registered in this Scope",
            "purpose": "the resolve purpose must be generation or audit",
            "store-unavailable": "the Source lifecycle store is not configured for this Runtime",
            "unavailable": "the node is not available for generation",
        }
        super().__init__(messages.get(code, f"invalid Source lifecycle request: {code}"))


class SourceLifecycleEvent(BaseModel):
    """One validated Source lifecycle transition request.

    The event names the Source node, the lifecycle revision it applies to, the
    per-Source event sequence, the availability the Source carries afterwards,
    and the reason for the transition. Extra fields are rejected so a caller
    cannot smuggle an unvalidated instruction into the event log.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    revision: StrictInt = Field(ge=1)
    sequence: StrictInt = Field(ge=1)
    available: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("node_id", "reason")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")  # noqa: TRY003
        return value


class ScopedSourceLifecycleApplication:
    """Register evidence dependencies and propagate Source availability in one Scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(self, node_id: str, source_ref: SourceRef, revision: int = 1) -> None:
        """Register one captured Source as the root of the evidence graph."""

        identifier = _node_id(node_id)
        _revision(revision)
        await self._require_readable_source(source_ref)
        reference = source_ref.model_dump_json()
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            nodes = await self._nodes()
            existing = nodes.get(identifier)
            if existing is None:
                node = SourceLifecycleNode(
                    node_id=identifier,
                    kind=SOURCE_NODE,
                    ordinal=len(nodes),
                    reference=reference,
                    parents=(),
                    policy=None,
                    revision=revision,
                    sequence=0,
                    available=True,
                    event_reason=None,
                )
                await self._store().register(self.scope_id, node)
                nodes[identifier] = node
            else:
                _require_same_binding(
                    existing,
                    kind=SOURCE_NODE,
                    reference=reference,
                    parents=(),
                    policy=None,
                    revision=revision,
                )
            await self._reconcile(context.artifacts.memory, nodes, "registered Source")

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        parents: Sequence[str] = (),
        policy: str = ALL_POLICY,
    ) -> None:
        """Register one readable Memory entry as a derived evidence node."""

        identifier = _node_id(node_id)
        parent_ids = _parent_ids(parents)
        _policy(policy)
        await self._require_readable_memory(citation)
        reference = citation.model_dump_json()
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            nodes = await self._nodes()
            existing = nodes.get(identifier)
            if existing is None:
                _require_known_parents(nodes, identifier, parent_ids)
                node = SourceLifecycleNode(
                    node_id=identifier,
                    kind=MEMORY_NODE,
                    ordinal=len(nodes),
                    reference=reference,
                    parents=parent_ids,
                    policy=policy,
                    revision=None,
                    sequence=0,
                    available=False,
                    event_reason=None,
                )
                await self._store().register(self.scope_id, node)
                nodes[identifier] = node
            else:
                _require_same_binding(
                    existing,
                    kind=MEMORY_NODE,
                    reference=reference,
                    parents=parent_ids,
                    policy=policy,
                    revision=None,
                )
            await self._reconcile(context.artifacts.memory, nodes, "registered Memory")

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        parents: Sequence[str] = (),
        policy: str = ALL_POLICY,
    ) -> None:
        """Register one readable Topic Memory Artifact as a derived evidence node."""

        identifier = _node_id(node_id)
        parent_ids = _parent_ids(parents)
        _policy(policy)
        if artifact_ref.family != TopicMemory.family:
            raise SourceLifecycleError("invalid-family")
        await self._require_readable_topic(artifact_ref)
        reference = artifact_ref.model_dump_json()
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            nodes = await self._nodes()
            existing = nodes.get(identifier)
            if existing is None:
                _require_known_parents(nodes, identifier, parent_ids)
                node = SourceLifecycleNode(
                    node_id=identifier,
                    kind=TOPIC_NODE,
                    ordinal=len(nodes),
                    reference=reference,
                    parents=parent_ids,
                    policy=policy,
                    revision=None,
                    sequence=0,
                    available=False,
                    event_reason=None,
                )
                await self._store().register(self.scope_id, node)
                nodes[identifier] = node
            else:
                _require_same_binding(
                    existing,
                    kind=TOPIC_NODE,
                    reference=reference,
                    parents=parent_ids,
                    policy=policy,
                    revision=None,
                )
            await self._reconcile(context.artifacts.memory, nodes, "registered Topic Memory")

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one Source lifecycle event and return every available node."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("invalid-event")
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            return await self._apply(context.artifacts.memory, event)

    async def resolve(
        self,
        node_id: str,
        purpose: str = GENERATION_PURPOSE,
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        """Resolve one registered node for generation or for audit."""

        identifier = _node_id(node_id)
        if purpose not in (GENERATION_PURPOSE, AUDIT_PURPOSE):
            raise SourceLifecycleError("purpose")
        async with self._runtime._scoped_operation(self.scope_id):
            nodes = await self._nodes()
            node = nodes.get(identifier)
            if node is None:
                raise SourceLifecycleError("not-registered")
            if purpose == GENERATION_PURPOSE and not _availability(nodes).get(identifier, False):
                raise SourceLifecycleError("unavailable")
        if node.kind == SOURCE_NODE:
            return SourceRef.model_validate_json(node.reference)
        if node.kind == MEMORY_NODE:
            return MemoryCitation.model_validate_json(node.reference)
        return ArtifactRef.model_validate_json(node.reference)

    async def unavailable_topic_artifacts(self) -> frozenset[str]:
        """Return the Topic Artifact identities that lost their support."""

        nodes = await self._nodes()
        availability = _availability(nodes)
        artifacts = set()
        for node in nodes.values():
            if node.kind != TOPIC_NODE or availability.get(node.node_id, True):
                continue
            artifacts.add(ArtifactRef.model_validate_json(node.reference).artifact_id)
        return frozenset(artifacts)

    async def _apply(self, service: MemoryService, event: SourceLifecycleEvent) -> list[str]:
        nodes = await self._nodes()
        source = nodes.get(event.node_id)
        if source is None or source.kind != SOURCE_NODE:
            raise SourceLifecycleError("not-a-source")
        if source.revision != event.revision:
            raise SourceLifecycleError("event-revision")
        if event.sequence < source.sequence:
            raise SourceLifecycleError("event-sequence")
        if event.sequence == source.sequence:
            if source.event_reason != event.reason or source.available != event.available:
                raise SourceLifecycleError("event-conflict")
        else:
            updated = replace(
                source,
                sequence=event.sequence,
                available=event.available,
                event_reason=event.reason,
            )
            nodes[event.node_id] = updated
            await self._store().record_event(
                self.scope_id,
                event.node_id,
                sequence=event.sequence,
                available=event.available,
                reason=event.reason,
                states=_availability(nodes),
            )
        await self._reconcile(service, nodes, event.reason)
        return sorted(node_id for node_id, state in _availability(nodes).items() if state)

    async def _reconcile(
        self,
        service: MemoryService,
        nodes: Mapping[str, SourceLifecycleNode],
        reason: str,
    ) -> None:
        """Make the native Memory entry states match the derived availability."""

        availability = _availability(nodes)
        targets: dict[str, dict[str, bool]] = {}
        for node in nodes.values():
            if node.kind != MEMORY_NODE:
                continue
            citation = MemoryCitation.model_validate_json(node.reference)
            entries = targets.setdefault(citation.memory_ref.artifact_id, {})
            entries.setdefault(citation.entry_id, False)
            if availability.get(node.node_id, True):
                # One body may carry several nodes. Any node that still holds
                # support keeps the body usable, so an unrelated withdrawal
                # never removes a fact that another node still supports.
                entries[citation.entry_id] = True
        for artifact_id, desired in targets.items():
            await self._reconcile_artifact(service, artifact_id, desired, reason)

    async def _reconcile_artifact(
        self,
        service: MemoryService,
        artifact_id: str,
        desired: Mapping[str, bool],
        reason: str,
    ) -> None:
        try:
            memory = await service.head(artifact_id)
        except ArtifactNotFoundError:
            return
        entries = await service.entries(memory)
        states = {item.entry_id: item.state for item in memory.content.manifest.entries}
        deactivate = [
            entry
            for entry in entries
            if entry.entry_id in desired and states.get(entry.entry_id) == "active" and not desired[entry.entry_id]
        ]
        reactivate = [
            entry
            for entry in entries
            if entry.entry_id in desired and states.get(entry.entry_id) == "inactive" and desired[entry.entry_id]
        ]
        if not deactivate and not reactivate:
            return
        note = f"Source lifecycle: {reason}"[:_MAX_MEMORY_REASON_LENGTH]
        if deactivate:
            memory = await service.forget(memory, entries=deactivate, reason=note)
        if reactivate:
            await service.reactivate(memory, entries=reactivate, reason=note)

    async def _nodes(self) -> dict[str, SourceLifecycleNode]:
        nodes = await self._store().nodes(self.scope_id)
        return {node.node_id: node for node in nodes}

    def _store(self) -> SourceLifecycleStore:
        return self._runtime.lifecycle.store()

    async def _require_readable_source(self, source_ref: SourceRef) -> None:
        try:
            await self._runtime._records().get_source(
                self.scope_id,
                source_ref.source_type,
                source_ref.source_id,
            )
        except (BaseValueNotFoundError, InvalidBaseAccessRequestError) as error:
            raise SourceLifecycleError("missing-reference") from error

    async def _require_readable_memory(self, citation: MemoryCitation) -> None:
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        except Exception as error:
            raise SourceLifecycleError("missing-reference") from error

    async def _require_readable_topic(self, artifact_ref: ArtifactRef) -> None:
        try:
            await self._runtime._records().get_artifact_revision(
                self.scope_id,
                artifact_ref.family,
                artifact_ref.artifact_id,
                artifact_ref.revision,
            )
        except (BaseValueNotFoundError, InvalidBaseAccessRequestError) as error:
            raise SourceLifecycleError("missing-reference") from error


class SourceLifecycleApplication:
    """Select the scoped Source lifecycle application service."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime
        self._store: SourceLifecycleStore | None = None

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)

    def store(self) -> SourceLifecycleStore:
        """Resolve the durable store behind the configured provider."""

        if self._store is None:
            # The relational provider owns the database this Runtime reads and
            # writes; a provider without one simply has no lifecycle store.
            database = getattr(self._runtime._provider, "database", None)
            if not isinstance(database, AsyncDatabase):
                raise SourceLifecycleError("store-unavailable")
            self._store = SourceLifecycleStore(database)
        return self._store

    async def filter_topic_hits(
        self,
        scope_id: str,
        result: TopicMemorySearchResult,
    ) -> TopicMemorySearchResult:
        """Drop the Topic Memory hits whose node lost its support."""

        try:
            unavailable = await self.for_scope(scope_id).unavailable_topic_artifacts()
        except Exception as error:
            log_safely(
                logger,
                logging.ERROR,
                "Source lifecycle Topic filtering failed",
                exc_info=error,
                extra={
                    "event": "source_lifecycle.topic_filter.failed",
                    "outcome": "failure",
                    "unit": "topic-memory",
                },
            )
            return result
        if not unavailable:
            return result
        hits = tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in unavailable)
        if len(hits) == len(result.hits):
            return result
        return result.model_copy(update={"hits": hits})


def _availability(nodes: Mapping[str, SourceLifecycleNode]) -> dict[str, bool]:
    """Compute every node's availability from the Source states and the policies."""

    available: dict[str, bool] = {}
    for node in nodes.values():
        if node.kind == SOURCE_NODE:
            available[node.node_id] = node.available
            continue
        support = [available.get(parent, False) for parent in node.parents]
        available[node.node_id] = all(support) if node.policy == ALL_POLICY else any(support)
    return available


def _node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("invalid-node-id")
    return value


def _revision(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SourceLifecycleError("invalid-revision")
    return value


def _policy(value: str) -> str:
    if value not in (ALL_POLICY, ANY_POLICY):
        raise SourceLifecycleError("invalid-policy")
    return value


def _parent_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SourceLifecycleError("no-parents")
    try:
        parents = tuple(values)
    except TypeError as error:
        raise SourceLifecycleError("no-parents") from error
    if not parents:
        raise SourceLifecycleError("no-parents")
    if any(not isinstance(parent, str) or not parent.strip() for parent in parents):
        raise SourceLifecycleError("invalid-node-id")
    if len(set(parents)) != len(parents):
        raise SourceLifecycleError("duplicate-parent")
    return parents


def _require_same_binding(
    existing: SourceLifecycleNode,
    *,
    kind: str,
    reference: str,
    parents: tuple[str, ...],
    policy: str | None,
    revision: int | None,
) -> None:
    if (
        existing.kind != kind
        or existing.reference != reference
        or existing.parents != parents
        or existing.policy != policy
        or existing.revision != revision
    ):
        raise SourceLifecycleError("conflict")


def _require_known_parents(
    nodes: Mapping[str, SourceLifecycleNode],
    node_id: str,
    parents: tuple[str, ...],
) -> None:
    for parent in parents:
        if parent == node_id:
            raise SourceLifecycleError("cycle")
        if parent not in nodes:
            raise SourceLifecycleError("missing-parent")
    if _reaches(nodes, node_id, parents):
        raise SourceLifecycleError("cycle")


def _reaches(nodes: Mapping[str, SourceLifecycleNode], node_id: str, parents: tuple[str, ...]) -> bool:
    """Report whether any parent already depends on the node being registered."""

    seen: set[str] = set()
    pending = list(parents)
    while pending:
        current = pending.pop()
        if current == node_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        parent = nodes.get(current)
        if parent is not None:
            pending.extend(parent.parents)
    return False


__all__ = [
    "ScopedSourceLifecycleApplication",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
]
