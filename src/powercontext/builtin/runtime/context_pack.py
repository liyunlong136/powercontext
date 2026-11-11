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

"""Wire protocol for sending one prepared context incrementally.

A Context Pack carries a complete prepared context: the body split into
bounded chunks, a digest over those chunks, and the pack's own byte count.
A Context Pack Delta carries only what changed: the number of leading base
chunks the receiver keeps, plus the suffix it has to append. A receiver that
holds the base pack therefore rebuilds the round's context without receiving
the unchanged prefix again.

The wire form is an interoperability contract. Field names, the canonical
JSON encoding, the Base64 chunk encoding, the digest format, and the byte
count semantics are fixed by the ``powercontext.context-pack.v1`` and
``powercontext.context-pack-delta.v1`` schemas, so a receiver can decode a
pack without sharing any state with the sender.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from powercontext.errors import PowerContextError

CONTEXT_PACK_SCHEMA: Final[str] = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA: Final[str] = "powercontext.context-pack-delta.v1"
DEFAULT_CONTEXT_CHUNK_BYTES: Final[int] = 512

_PACK_FIELDS: Final[frozenset[str]] = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
)

# The byte count is part of the payload it measures, so the length has to be
# resolved against itself. Three passes always settle it: the digit count of
# the candidate only changes when the candidate crosses a power of ten.
_MAX_BYTE_COUNT_PASSES: Final[int] = 4


class ContextPackError(PowerContextError, ValueError):
    """Raised when a Context Pack or delta violates the wire contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        messages = {
            "append-type": "appended chunks must be bytes",
            "base-digest-mismatch": "the delta does not belong to this base pack",
            "chunk-boundary": "the chunk limit cannot hold a whole UTF-8 character",
            "chunk-limit": "the chunk limit must be a positive integer",
            "content-encoding": "prepared content must be valid UTF-8",
            "content-type": "prepared content must be text or bytes",
            "delta-appended": "appended must be a list of Base64 chunks",
            "delta-digest": "a delta digest must be a non-empty string",
            "delta-fields": "a delta must contain exactly the schema, digest, prefix, and suffix fields",
            "delta-json": "a delta must be a JSON object",
            "delta-payload": "a delta must be UTF-8 encoded JSON",
            "delta-prefix": "prefix_items must be a non-negative integer",
            "delta-schema": "a delta must use the context pack delta schema",
            "item-type": "context pack chunks must be bytes",
            "pack-byte-count": "the declared byte count does not match the serialized pack",
            "pack-digest": "the declared digest does not match the pack contents",
            "pack-fields": "a pack must contain exactly the schema, items, digest, and byte count fields",
            "pack-items": "items must be a list of Base64 chunks",
            "pack-json": "a pack must be a JSON object",
            "pack-payload": "a pack must be UTF-8 encoded JSON",
            "pack-schema": "a pack must use the context pack schema",
            "prefix-out-of-range": "prefix_items exceeds the number of base chunks",
            "target-digest-mismatch": "the rebuilt pack does not match the target digest",
        }
        super().__init__(messages.get(code, f"invalid Context Pack: {code}"))


@dataclass(frozen=True)
class ContextPack:
    """One complete prepared context, split into bounded wire chunks."""

    items: tuple[bytes, ...]

    @classmethod
    def build(cls, items: Iterable[bytes]) -> ContextPack:
        """Build a complete pack from already split chunks."""

        return cls(items=_normalize_items(items, "item-type"))

    @classmethod
    def from_content(
        cls,
        content: str | bytes | None,
        *,
        chunk_bytes: int = DEFAULT_CONTEXT_CHUNK_BYTES,
    ) -> ContextPack:
        """Split prepared content into deterministic chunks that respect the caller's limit.

        The chunks keep the original bytes, so joining them reproduces the
        content exactly. Every chunk is valid UTF-8 on its own and never
        exceeds ``chunk_bytes``. A limit too small to hold a single character
        cannot produce a legal boundary and fails instead of splitting one.
        """

        return cls(items=_chunk_content(_encode_content(content), chunk_bytes))

    @property
    def digest(self) -> str:
        """The digest of this pack's schema and chunks."""

        return _digest_payload({"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(self.items)})

    @property
    def byte_count(self) -> int:
        """The serialized size of this pack, including the byte count field itself."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this pack into its canonical wire form."""

        payload: dict[str, Any] = {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": _encode_items(self.items),
            "digest": self.digest,
            "byte_count": 0,
        }
        count = 0
        for _ in range(_MAX_BYTE_COUNT_PASSES):
            payload["byte_count"] = count
            data = _canonical_json(payload)
            if len(data) == count:
                return data
            count = len(data)
        raise ContextPackError("pack-byte-count")

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode one pack from its wire form and verify its digest and byte count."""

        payload = _decode_payload(data, "pack")
        _require_fields(payload, _PACK_FIELDS, "pack-fields")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("pack-schema")
        items = _decode_items(payload["items"], "pack-items")
        digest = _require_digest(payload["digest"], "pack-digest")
        byte_count = _require_int(payload["byte_count"], "pack-byte-count")
        pack = cls(items=items)
        if pack.digest != digest:
            raise ContextPackError("pack-digest")
        if byte_count != len(_canonical_json(payload)):
            raise ContextPackError("pack-byte-count")
        return pack


@dataclass(frozen=True)
class ContextPackDelta:
    """The incremental form of one prepared context."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Describe how the receiver turns ``base`` into ``target``.

        The delta keeps the longest common prefix of the two chunk sequences
        and carries the target's remaining suffix, so the receiver rebuilds
        the target's chunks, order, body, and digest.
        """

        prefix = _common_prefix(base.items, target.items)
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """The serialized size of this delta."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this delta into its canonical wire form."""

        payload: dict[str, Any] = {
            "schema": CONTEXT_PACK_DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode one delta from its wire form."""

        payload = _decode_payload(data, "delta")
        _require_fields(payload, _DELTA_FIELDS, "delta-fields")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("delta-schema")
        base_digest = _require_digest(payload["base_digest"], "delta-digest")
        target_digest = _require_digest(payload["target_digest"], "delta-digest")
        prefix_items = _require_prefix(payload["prefix_items"])
        appended = _decode_items(payload["appended"], "delta-appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Apply one delta to a base pack and return the rebuilt target pack.

    The base digest, the prefix bound, and the rebuilt digest are all checked,
    so a delta applied to the wrong base, with an out-of-range prefix, or with a
    target digest that does not match the rebuilt chunks fails instead of
    returning a context the sender never produced.
    """

    if delta.base_digest != base.digest:
        raise ContextPackError("base-digest-mismatch")
    prefix = _require_prefix(delta.prefix_items)
    if prefix > len(base.items):
        raise ContextPackError("prefix-out-of-range")
    target = ContextPack(items=base.items[:prefix] + _normalize_items(delta.appended, "append-type"))
    if target.digest != delta.target_digest:
        raise ContextPackError("target-digest-mismatch")
    return target


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_payload(payload: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _encode_items(items: Iterable[bytes]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(values: object, code: str) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError(code)
    decoded: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError(code)
        try:
            decoded.append(base64.b64decode(value, validate=True))
        except ValueError:
            raise ContextPackError(code) from None
    return tuple(decoded)


def _normalize_items(items: Iterable[bytes], code: str) -> tuple[bytes, ...]:
    if isinstance(items, (str, bytes, bytearray, memoryview)):
        raise ContextPackError(code)
    normalized: list[bytes] = []
    for item in items:
        if isinstance(item, bytes):
            normalized.append(item)
        elif isinstance(item, (bytearray, memoryview)):
            normalized.append(bytes(item))
        else:
            raise ContextPackError(code)
    return tuple(normalized)


def _encode_content(content: str | bytes | None) -> bytes:
    if content is None:
        return b""
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, (bytes, bytearray, memoryview)):
        data = bytes(content)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise ContextPackError("content-encoding") from None
        return data
    raise ContextPackError("content-type")


def _chunk_content(data: bytes, limit: int) -> tuple[bytes, ...]:
    """Split content into chunks that respect ``limit`` without splitting a character.

    Line boundaries are preferred because they stay stable when the content
    grows at its end; a line longer than the limit is split at the widest
    character boundary the limit allows.
    """

    if type(limit) is not int or limit <= 0:
        raise ContextPackError("chunk-limit")
    chunks: list[bytes] = []
    current = bytearray()
    for line in data.splitlines(keepends=True):
        if len(current) + len(line) <= limit:
            current += line
            continue
        if current:
            chunks.append(bytes(current))
            current = bytearray()
        if len(line) <= limit:
            current = bytearray(line)
            continue
        pieces = _split_utf8(line, limit)
        chunks.extend(pieces[:-1])
        current = bytearray(pieces[-1])
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


def _split_utf8(data: bytes, limit: int) -> list[bytes]:
    pieces: list[bytes] = []
    index = 0
    total = len(data)
    while index < total:
        end = min(index + limit, total)
        while end > index and end < total and (data[end] & 0xC0) == 0x80:
            end -= 1
        if end == index:
            raise ContextPackError("chunk-boundary")
        pieces.append(data[index:end])
        index = end
    return pieces


def _common_prefix(base: tuple[bytes, ...], target: tuple[bytes, ...]) -> int:
    limit = min(len(base), len(target))
    index = 0
    while index < limit and base[index] == target[index]:
        index += 1
    return index


def _decode_payload(data: bytes, kind: str) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContextPackError(f"{kind}-payload")
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        raise ContextPackError(f"{kind}-payload") from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise ContextPackError(f"{kind}-json") from None
    if not isinstance(payload, dict):
        raise ContextPackError(f"{kind}-json")
    return payload


def _require_fields(payload: Mapping[str, Any], expected: frozenset[str], code: str) -> None:
    if set(payload) != expected:
        raise ContextPackError(code)


def _require_digest(value: object, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContextPackError(code)
    return value


def _require_int(value: object, code: str) -> int:
    if type(value) is not int:
        raise ContextPackError(code)
    return value


def _require_prefix(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ContextPackError("delta-prefix")
    return value


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CONTEXT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
