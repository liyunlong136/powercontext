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

"""Branch-exact capture of the visible history of one Pi session."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME: Final[str] = "pi-event"
PI_EVENT_SOURCE_VERSION: Final[str] = "1"
PI_SESSION_CONNECTOR_NAME: Final[str] = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION: Final[str] = "1"

PI_SESSION_FORMAT_VERSION: Final[int] = 3
PI_SESSION_CHECKPOINT_VERSION: Final[int] = 1

_UTF8_BOM: Final[str] = "\ufeff"
_CUSTOM_ROLE: Final[str] = "custom"
_SUMMARY_TYPES: Final[frozenset[str]] = frozenset({"compaction", "branch_summary"})


class PiSessionInput(BaseModel):
    """One offline Pi session archive together with the selected end of its history."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event captured from the selected branch of a Pi session."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("session_id", "event_id", "event_type", "role")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Pi event identity parts must not be empty")  # noqa: TRY003
        return value


class PiEventSource(Source):
    """A captured Pi event addressed by a stable session-and-event identity."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not value:
            raise ValueError("source_id must not be empty")  # noqa: TRY003
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Return the stable identity of one visible event inside one session.

    The identity only depends on the session identifier and the event identifier,
    so the same event keeps one address across branches, appends and restarts.
    """

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must not be empty")  # noqa: TRY003
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must not be empty")  # noqa: TRY003
    source_id = f"{PI_EVENT_SOURCE_NAME}:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(source_id) <= MAX_SOURCE_ID_LENGTH:
        return source_id
    digest = hashlib.sha256(f"{session_id}\0{event_id}".encode()).hexdigest()
    return f"{PI_EVENT_SOURCE_NAME}:{digest}"


@dataclass(frozen=True, slots=True)
class _SessionArchive:
    """One parsed session archive with its selected branch and its raw records."""

    session_id: str
    events: tuple[PiEventInput, ...]
    records: tuple[dict[str, JsonValue], ...]


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse one Pi session archive into the visible events of the selected branch.

    The archive is parsed and validated as a whole, so a malformed session fails
    atomically instead of returning the events of a partially understood history.
    """

    return _parse_archive(value).events


def _parse_archive(value: PiSessionInput) -> _SessionArchive:
    if not isinstance(value, PiSessionInput):
        raise ValueError("value must be a PiSessionInput")  # noqa: TRY003, TRY004
    records = _read_records(_decode_archive(value.content))
    session_id = _read_header(records[0])
    nodes = _read_nodes(records[1:])
    by_id: dict[str, dict[str, JsonValue]] = {node["id"]: node for node in nodes}
    leaf_id = _select_leaf(value.leaf_id, nodes, by_id)
    chain = _read_chain(leaf_id, by_id) if leaf_id is not None else ()
    events: list[PiEventInput] = []
    for record in chain:
        event = _build_event(session_id, record)
        if event is not None:
            events.append(event)
    return _SessionArchive(session_id=session_id, events=tuple(events), records=tuple(nodes))


def _decode_archive(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise ValueError("Pi session content must be bytes")  # noqa: TRY003, TRY004
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error  # noqa: TRY003
    if text.startswith(_UTF8_BOM):
        text = text[len(_UTF8_BOM) :]
    return text


def _read_records(text: str) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session line {number} is not valid JSON") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError(f"Pi session line {number} must be a JSON object")  # noqa: TRY003, TRY004
        records.append(record)
    if not records:
        raise ValueError("Pi session must contain a session header")  # noqa: TRY003
    return records


def _read_header(header: dict[str, JsonValue]) -> str:
    if header.get("type") != "session":
        raise ValueError("Pi session must start with a session header")  # noqa: TRY003
    version = header.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != PI_SESSION_FORMAT_VERSION:
        raise ValueError("Pi session header must declare version 3")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must declare a non-empty id")  # noqa: TRY003
    return session_id


def _read_nodes(records: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    nodes: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for record in records:
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("every Pi session record must declare a non-empty id")  # noqa: TRY003
        if node_id in seen:
            raise ValueError(f"Pi session record {node_id!r} is duplicated")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError(f"Pi session record {node_id!r} must declare parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip()):
            raise ValueError(f"Pi session record {node_id!r} must declare a string or null parentId")  # noqa: TRY003
        if parent_id and parent_id not in seen:
            raise ValueError(f"Pi session record {node_id!r} references an unknown parent")  # noqa: TRY003
        seen.add(node_id)
        nodes.append(record)
    return nodes


def _select_leaf(
    leaf_id: str | None,
    nodes: list[dict[str, JsonValue]],
    by_id: dict[str, dict[str, JsonValue]],
) -> str | None:
    if leaf_id is None:
        return nodes[-1]["id"] if nodes else None
    if leaf_id not in by_id:
        raise ValueError(f"Pi session leaf {leaf_id!r} does not exist")  # noqa: TRY003
    return leaf_id


def _read_chain(leaf_id: str, by_id: dict[str, dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    chain: list[dict[str, JsonValue]] = []
    visited: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in visited:
            raise ValueError("Pi session history contains a cycle")  # noqa: TRY003
        visited.add(current)
        record = by_id[current]
        chain.append(record)
        parent_id = record["parentId"]
        current = parent_id if isinstance(parent_id, str) and parent_id else None
    chain.reverse()
    return chain


def _build_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    if event_type == "message":
        rendered = _render_message(record)
    elif event_type in _SUMMARY_TYPES:
        rendered = _render_summary(event_type, record)
    elif event_type == "custom_message":
        rendered = _render_custom(record)
    else:
        return None
    if rendered is None:
        return None
    role, body = rendered
    parent_id = record["parentId"]
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=parent_id if isinstance(parent_id, str) and parent_id else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _render_message(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    if role == "toolResult":
        body = _render_tool_result(message)
    elif role == "bashExecution":
        body = _render_bash_execution(message)
    else:
        body = _render_blocks(message.get("content"), allow_tool_call=True)
    if not body:
        return None
    return role, body


def _render_summary(event_type: str, record: dict[str, JsonValue]) -> tuple[str, str] | None:
    summary = record.get("summary")
    if not isinstance(summary, str) or not summary:
        return None
    return event_type, summary


def _render_custom(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    body = _render_blocks(record.get("content"), allow_tool_call=False)
    if not body:
        return None
    return _CUSTOM_ROLE, body


def _render_tool_result(message: dict[str, JsonValue]) -> str:
    prefix = f"[tool_result {_text(message.get('toolName'))} {_text(message.get('toolCallId'))}]"
    body = _render_blocks(message.get("content"), allow_tool_call=False)
    return f"{prefix} {body}" if body else prefix


def _render_bash_execution(message: dict[str, JsonValue]) -> str:
    command = message.get("command")
    output = message.get("output")
    command_text = command if isinstance(command, str) else ""
    output_text = output if isinstance(output, str) else ""
    if not command_text and not output_text:
        return ""
    head = f"$ {command_text}"
    return f"{head}\n{output_text}" if output_text else head


def _render_blocks(content: JsonValue, /, *, allow_tool_call: bool) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall" and allow_tool_call:
            parts.append(_render_tool_call(block))
    return "\n".join(parts)


def _render_tool_call(block: dict[str, JsonValue]) -> str:
    arguments = block.get("arguments")
    payload = arguments if isinstance(arguments, dict) else {}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"[tool_call {_text(block.get('name'))} {_text(block.get('id'))}] {encoded}"


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _record_checksum(record: Mapping[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiEventSourceAdapter:
    """Resolve and read the event-level Pi Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one captured Pi event as standard textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the visible events of one selected branch of a Pi session."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Import every visible event of the selected branch that is not captured yet.

        The archive is parsed completely before anything is submitted, so a broken
        input never advances the durable progress of the binding.
        """

        archive = _parse_archive(PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id))
        imported = _load_imported(session.checkpoint)
        _reject_rewritten_events(archive, imported)
        added: dict[str, str] = {}
        for event in archive.events:
            source_id = pi_event_source_id(event.session_id, event.event_id)
            if source_id in imported:
                continue
            outcome = await session.submit(source_id, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is ConnectorSubmissionStatus.ACCEPTED:
                added[source_id] = _record_checksum(event.record)
        if not added:
            return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=session.checkpoint)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": PI_SESSION_CHECKPOINT_VERSION, "imported": {**imported, **added}},
        )


def _load_imported(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")  # noqa: TRY003, TRY004
    version = checkpoint.get("version")
    if version is not None and version != PI_SESSION_CHECKPOINT_VERSION:
        raise ValueError("Pi session checkpoint version is not supported")  # noqa: TRY003
    imported = checkpoint.get("imported", {})
    if not isinstance(imported, dict):
        raise ValueError("Pi session checkpoint must map imported identities to checksums")  # noqa: TRY003, TRY004
    entries: dict[str, str] = {}
    for source_id, checksum in imported.items():
        if not isinstance(source_id, str) or not isinstance(checksum, str):
            raise ValueError("Pi session checkpoint must map imported identities to checksums")  # noqa: TRY003, TRY004
        entries[source_id] = checksum
    return entries


def _reject_rewritten_events(archive: _SessionArchive, imported: dict[str, str]) -> None:
    if not imported:
        return
    for record in archive.records:
        event_id = record.get("id")
        if not isinstance(event_id, str):
            continue
        stored = imported.get(pi_event_source_id(archive.session_id, event_id))
        if stored is None:
            continue
        if stored != _record_checksum(record):
            raise ValueError(f"Pi session event {event_id!r} was rewritten after it was imported")  # noqa: TRY003


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CHECKPOINT_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PI_SESSION_FORMAT_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
