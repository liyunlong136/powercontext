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

"""Row-level CSV capture for table-shaped offline evidence."""

from __future__ import annotations

import csv
import io
import json
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel, field_validator, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME: Final[str] = "csv-row"
CSV_ROW_SOURCE_VERSION: Final[str] = "1"

_UTF8_BOM: Final[str] = "\ufeff"


class CSVDocumentInput(BaseModel):
    """One offline CSV document captured as a table."""

    document_id: str
    primary_key: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value:
            raise ValueError("document_id must not be empty")  # noqa: TRY003
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must not be empty")  # noqa: TRY003
        return value


class CSVRowInput(BaseModel):
    """One parsed CSV row that keeps its original column order and values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value:
            raise ValueError("document_id must not be empty")  # noqa: TRY003
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must not be empty")  # noqa: TRY003
        return value

    @field_validator("primary_key_value")
    @classmethod
    def validate_primary_key_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("primary_key_value must not be empty or whitespace-only")  # noqa: TRY003
        return value

    @model_validator(mode="after")
    def validate_row_shape(self) -> CSVRowInput:
        if not self.columns:
            raise ValueError("columns must not be empty")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("values must provide exactly one value per column")  # noqa: TRY003
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")  # noqa: TRY003
        return self


class CSVRowSource(Source):
    """A captured CSV row addressed by a stable document-and-key identity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not value:
            raise ValueError("source_id must not be empty")  # noqa: TRY003
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value:
            raise ValueError("document_id must not be empty")  # noqa: TRY003
        if value != value.strip():
            raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
        return value

    @field_validator("primary_key")
    @classmethod
    def validate_primary_key(cls, value: str) -> str:
        if not value:
            raise ValueError("primary_key must not be empty")  # noqa: TRY003
        return value

    @field_validator("primary_key_value")
    @classmethod
    def validate_primary_key_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("primary_key_value must not be empty or whitespace-only")  # noqa: TRY003
        return value

    @model_validator(mode="after")
    def validate_row_shape(self) -> CSVRowSource:
        if not self.columns:
            raise ValueError("columns must not be empty")  # noqa: TRY003
        if len(self.columns) != len(self.values):
            raise ValueError("values must provide exactly one value per column")  # noqa: TRY003
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must be unique")  # noqa: TRY003
        return self


def csv_row_source_id(document_id: str, primary_key_value: str, /) -> str:
    """Return the stable identity of one row inside one document."""

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must not be empty")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must not be empty or whitespace-only")  # noqa: TRY003
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def parse_csv_document(value: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse one offline CSV document into validated row-level inputs.

    The whole document is parsed and validated before any row is returned, so a
    malformed document fails atomically instead of yielding partial rows.
    """

    text = _decode_document(value.content)
    records = _read_records(text)
    if not records:
        raise ValueError("CSV document must contain a header record")  # noqa: TRY003
    header = records[0]
    if not header:
        raise ValueError("CSV document must contain a header record")  # noqa: TRY003
    _validate_header(header)
    if value.primary_key not in header:
        raise ValueError("primary_key must match a header column exactly")  # noqa: TRY003

    columns = tuple(header)
    primary_key_index = columns.index(value.primary_key)
    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("CSV record width must match the header width")  # noqa: TRY003
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key value must not be empty or whitespace-only")  # noqa: TRY003
        if primary_key_value in seen:
            raise ValueError("primary key values must be unique within one document")  # noqa: TRY003
        seen.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def _decode_document(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise ValueError("CSV content must be bytes")  # noqa: TRY003, TRY004
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error  # noqa: TRY003
    if text.startswith(_UTF8_BOM):
        text = text[len(_UTF8_BOM) :]
    return text


def _read_records(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        return list(reader)
    except csv.Error as error:
        raise ValueError(f"CSV content is malformed: {error}") from error  # noqa: TRY003


def _validate_header(header: list[str]) -> None:
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("header field names must not be empty or whitespace-only")  # noqa: TRY003
        if column in seen:
            raise ValueError("header field names must be unique")  # noqa: TRY003
        seen.add(column)


class CSVRowSourceAdapter:
    """Resolve and read the row-level CSV Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one CSV row as standard textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = _row_mapping(source)
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


def _row_mapping(source: CSVRowSource) -> dict[str, str]:
    return dict(zip(source.columns, source.values, strict=True))


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
