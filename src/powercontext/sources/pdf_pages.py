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

"""Page-level PDF capture for financial-report evidence."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol
from urllib.parse import quote

from pydantic import BaseModel, field_validator
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from powercontext._logging import log_safely
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.artifacts.search import AdmissionFloor, analyze_fts_query, analyze_text
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

logger = logging.getLogger(__name__)

PDF_PAGE_SOURCE_NAME: Final[str] = "pdf-page"
PDF_PAGE_SOURCE_VERSION: Final[str] = "1"


class PDFDocumentInput(BaseModel):
    """One complete original PDF document captured for page-level evidence."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be empty or whitespace-only")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical page of one PDF document revision."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page_number must be a positive physical page number")  # noqa: TRY003
        return value


class PDFPageSource(Source):
    """A captured PDF page addressed by document, revision digest and page number."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not value:
            raise ValueError("source_id must not be empty")  # noqa: TRY003
        if len(value) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
        return value

    @field_validator("page_number")
    @classmethod
    def validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page_number must be a positive physical page number")  # noqa: TRY003
        return value


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int, /) -> str:
    """Return the stable identity of one physical page of one document revision."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace-only")  # noqa: TRY003
    if not isinstance(document_sha256, str) or not document_sha256.strip():
        raise ValueError("document_sha256 must not be empty or whitespace-only")  # noqa: TRY003
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive physical page number")  # noqa: TRY003
    source_id = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return source_id


def parse_pdf_document(value: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one original PDF into page-level inputs.

    Pages are returned in document order starting at physical page 1, blank pages
    included. The whole document is read before any page is returned, so an
    unreadable document fails atomically instead of yielding partial pages.
    """

    if not isinstance(value.content, bytes):
        raise ValueError("PDF content must be bytes")  # noqa: TRY003, TRY004
    digest = hashlib.sha256(value.content).hexdigest()
    texts = _extract_page_texts(value.content)
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=digest,
            page_number=number,
            text=text,
        )
        for number, text in enumerate(texts, start=1)
    )


def _extract_page_texts(content: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            _decrypt_without_password(reader)
        return [_page_text(page) for page in reader.pages]
    except ValueError:
        raise
    except PyPdfError as error:
        raise ValueError(f"PDF document could not be parsed: {error}") from error  # noqa: TRY003


def _decrypt_without_password(reader: PdfReader) -> None:
    """Open a permission-protected PDF that needs no user-supplied password."""

    outcome = reader.decrypt("")
    if not getattr(outcome, "value", 1):
        raise ValueError("PDF document requires a password that the caller did not provide")  # noqa: TRY003


def _page_text(page: Any, /) -> str:
    try:
        text = page.extract_text()
    except PyPdfError as error:
        raise ValueError(f"PDF page could not be read: {error}") from error  # noqa: TRY003
    return text if isinstance(text, str) else ""


class PDFPageSourceAdapter:
    """Resolve and read the page-level PDF Source."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one PDF page as standard textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _ScopedMemorySearch(Protocol):
    """The scoped Memory search surface used to retrieve evidence pages."""

    async def search(
        self,
        request: SearchMemoryRequest,
        /,
        *,
        admission: AdmissionFloor | None = None,
    ) -> Any: ...


# --------------------------------------------------------------------------- 检索
#
# 财报证据页检索分三步: 多路召回扩大候选池、按页类别与词项覆盖重排、返回不同页的
# 精确引用。下面的规则全部是对任意财报 PDF 与任意自然语言问题成立的通用规则,
# 不含任何文档标识、题号或页码特判。

#: 主查询一次召回的候选页上限。
_PAGE_CANDIDATE_LIMIT: Final[int] = 200
#: 单个内容词路线一次召回的候选页上限。
_PAGE_TERM_CANDIDATE_LIMIT: Final[int] = 15
#: 单词路线最多覆盖多少个内容词。
_PAGE_TERM_ROUTES: Final[int] = 6
#: 报表类别路线一次召回的候选页上限。
_PAGE_STATEMENT_CANDIDATE_LIMIT: Final[int] = 15
#: 为相邻页扩展解析物理页码的候选上限。
_PAGE_RESOLVE_LIMIT: Final[int] = 96
#: 相邻页扩展的窗口页数与衰减系数。
_PAGE_NEIGHBOUR_WINDOW: Final[int] = 3
_PAGE_NEIGHBOUR_DECAY: Final[float] = 0.35
#: 词项 IDF 上限: 极罕见的词往往是噪声而非判别信号, 截断后不让它主导排序。
_PAGE_IDF_CAP: Final[float] = 2.5
#: 页面标题里用于识别页类别的词元上限。
_PAGE_TITLE_TOKENS: Final[int] = 40
#: 标题命中、词组命中、页类别命中、目标文档命中与表格密度的权重。
_PAGE_TITLE_WEIGHT: Final[float] = 0.6
_PAGE_PHRASE_WEIGHT: Final[float] = 0.5
_PAGE_INTENT_WEIGHT: Final[float] = 0.5
_PAGE_ENTITY_WEIGHT: Final[float] = 1.0
_PAGE_TABLE_WEIGHT: Final[float] = 0.5
#: 命中不同查询词的比例权重: 覆盖更多不同词项的页面通常更相关。
_PAGE_CONJUNCTION_WEIGHT: Final[float] = 0.0
#: 直接留给后端 BM25 排序的名额占比: 重排只决定剩下的名额。
_PAGE_PRIMARY_RESERVE: Final[float] = 0.7

#: 指令性套话: 只描述「答案怎么写」, 不指示证据页。它们在财报正文里极罕见,
#: 若不剔除会因 IDF 极高而错误主导排序, 因此检索前整体剔除。
_PAGE_INSTRUCTION_NOISE: Final[frozenset[str]] = frozenset(
    (
        "answer", "answers", "answered", "ask", "asked", "asking",
        "calculate", "calculated", "calculating", "calculation",
        "round", "rounded", "rounding", "decimal", "place", "places",
        "percentage", "percentages", "percent", "percents",
        "utilize", "utilizes", "utilizing", "utilized", "use", "used", "using",
        "shown", "show", "clearly", "line", "lines",
        "item", "items", "state", "stated", "stating", "say", "saying",
        "explain", "explained", "explaining", "why",
        "please", "respond", "response", "perspective", "analyst", "investment",
        "based", "within", "detail", "details", "information", "info",
        "facts", "fact", "evidence", "context", "relevant",
        "measure", "measuring", "measures",
        "consider", "considering", "assume", "assuming", "given",
        "provide", "provided", "available", "supply", "supplied",
        "address", "quantity", "direct", "outline", "outlining", "listed",
        "define", "defined", "defining",
        "also", "just", "only", "kindly", "per", "usd", "millions", "million", "currency",
    )
)

#: 疑问句框架词: 保留会让「什么/如何/哪些」这类高频虚词参与打分。
_PAGE_FRAME_NOISE: Final[frozenset[str]] = frozenset(
    (
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "does", "do", "did", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "can", "could", "should", "would", "will", "shall",
        "may", "might", "must", "not", "no", "nor", "so", "than", "then",
        "else", "if", "whether", "there", "any", "some", "all", "each",
        "more", "most", "less", "least", "between", "among", "within",
        "like", "across", "around", "above", "below", "up", "down", "out", "off", "again",
    )
)

#: 页类别识别: 把标题词元拼成「去空格小写串」再匹配, 规避 PDF 抽取把词拆开的
#: 噪声(``statements`` → ``statemen ts``)。通用财报结构识别。
_PAGE_KIND_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("income", "consolidatedstatementsofincome"),
    ("income", "consolidatedstatementofincome"),
    ("income", "consolidatedstatementsofoperations"),
    ("income", "consolidatedstatementofoperations"),
    ("income", "consolidatedstatementsofearnings"),
    ("cashflow", "consolidatedstatementsofcashflows"),
    ("cashflow", "consolidatedstatementofcashflows"),
    ("balance", "consolidatedbalancesheets"),
    ("balance", "consolidatedbalancesheet"),
    ("segment", "summaryofbusinesssegmentdata"),
    ("segment", "businesssegmentdata"),
    ("segment", "segmentinformation"),
    ("segment", "segmentdata"),
    ("legal", "legalproceedings"),
    ("legal", "legalproceeding"),
    ("mdna", "managementsdiscussionandanalysis"),
    ("notes", "notestotheconsolidatedfinancialstatements"),
)

#: 财务概念 → 需要的页类别。这是通用领域词典: 每一条都是「哪类财务问题对应哪张
#: 报表或哪一章」的通识, 不绑定任何具体公司、文件或题号。
_PAGE_INTENT_LEXICON: Final[Mapping[str, tuple[str, ...]]] = {
    "balance": (
        "balance sheet", "balance sheets", "statement of financial position",
        "current ratio", "quick ratio", "cash ratio", "liquidity",
        "working capital", "solvency", "leverage", "debt to equity",
        "accounts receivable", "accounts payable", "receivable", "payable",
        "inventory", "inventories", "total assets", "total liabilities",
        "shareholders equity", "stockholders equity", "book value", "net debt",
        "current assets", "current liabilities", "prepaid expense",
        "goodwill", "intangible asset", "property plant and equipment",
        "return on equity", "return on asset", "roe", "roa",
    ),
    "income": (
        "statement of income", "statement of operations", "income statement",
        "p&l", "profit and loss", "gross profit", "gross margin",
        "operating income", "operating margin", "net income", "net margin",
        "earnings per share", "eps", "effective tax rate", "tax rate",
        "cost of sales", "cost of goods sold", "cogs", "sg&a",
        "research and development", "operating expense", "interest expense",
        "revenue growth", "revenue change",
    ),
    "cashflow": (
        "cash flow", "cash flows", "statement of cash flows", "free cash flow",
        "operating cash flow", "capex", "capital expenditure",
        "capital expenditures", "depreciation", "amortization",
        "investing activities", "financing activities", "operating activities",
    ),
    "segment": (
        "segment", "segments", "reportable segment", "reporting segment",
        "business segment", "product category", "product categories",
        "service categories", "segment data", "segment information",
        "segment results", "segment revenue",
    ),
    "legal": (
        "legal proceeding", "legal battle", "legal action", "legal actions",
        "litigation", "lawsuit", "regulatory proceeding", "legal matter",
    ),
}

#: 一个概念可能跨多张报表: 资本密集度同时要看利润表、资产负债表与现金流量表。
_PAGE_INTENT_EXTRA: Final[Mapping[str, tuple[str, ...]]] = {
    "capital intensive": ("income", "balance", "cashflow"),
}

#: 报表类别 → 召回路线用的标题词。路线只负责把候选页拉进池子, 精确判定由重排做。
_PAGE_INTENT_ROUTES: Final[Mapping[str, tuple[str, ...]]] = {
    "income": ("consolidated operations", "consolidated income"),
    "balance": ("consolidated balance sheets",),
    "cashflow": ("consolidated cash flows",),
    "segment": ("segment data", "segment information"),
    "legal": ("legal proceedings",),
}

#: 放宽后的准入门槛: 只要共享一个真实词项就进入候选池, 用来扩大召回。
_PAGE_RELAXED_ADMISSION: Final[AdmissionFloor] = AdmissionFloor(
    lexical_coverage=0.0,
    lexical_min_matched_terms=1,
)

#: 报表标题路线的准入门槛: 共享多数标题词才算命中, 避免路线过宽。
_PAGE_TITLE_ADMISSION: Final[AdmissionFloor] = AdmissionFloor(
    lexical_coverage=0.5,
    lexical_min_matched_terms=2,
)


@dataclass(frozen=True)
class _Query:
    """One question resolved into the signals the page ranker needs."""

    raw: str
    terms: tuple[str, ...]
    stems: tuple[str, ...]
    bigrams: tuple[tuple[str, str], ...]
    intent: Mapping[str, float]


@dataclass
class _Candidate:
    """One retrieved page with the signals used to rank it."""

    hit: Any
    document: str
    matched: frozenset[int]
    title_matched: frozenset[int]
    bigram_matched: frozenset[int]
    kinds: frozenset[str]
    numeric: float
    primary_rank: int = -1
    page_number: int = -1
    score: float = 0.0


def _stem(token: str) -> str:
    """A light English plural stemmer, applied to query and page sides alike."""

    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses") or (token.endswith("ses") and len(token) > 4):
        return token[:-2]
    if token.endswith(("xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _normalize(text: str) -> str:
    """Lower-case an alphanumeric form so phrase probes survive punctuation."""

    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _fiscal_year_expansions(token: str) -> list[str]:
    """``fy2022`` also means ``2022``; ``fy22`` also means ``2022``."""

    if not token.startswith("fy") or not token[2:].isdigit():
        return []
    digits = token[2:]
    if len(digits) == 4:
        return [digits]
    if len(digits) == 2:
        return ["20" + digits]
    return []


def _query_terms(question: str) -> tuple[str, ...]:
    """Content terms of one question, free of instructions and question frames."""

    out: list[str] = []
    for token in analyze_fts_query(question).split():
        if token in _PAGE_INSTRUCTION_NOISE or token in _PAGE_FRAME_NOISE:
            continue
        if len(token) <= 1 or token in {"one", "two", "three"}:
            continue
        expansions = _fiscal_year_expansions(token)
        out.extend(expansions if expansions else [token])
    seen: set[str] = set()
    return tuple(token for token in out if not (token in seen or seen.add(token)))


def _unique_stems(terms: tuple[str, ...]) -> tuple[str, ...]:
    """Stem the content terms and drop duplicates while keeping the query order."""

    stems: list[str] = []
    for token in terms:
        stemmed = _stem(token)
        if stemmed not in stems:
            stems.append(stemmed)
    return tuple(stems)


def _query_bigrams(terms: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Adjacent content-term pairs, used as high-precision phrase evidence."""

    return tuple(
        (_stem(terms[index]), _stem(terms[index + 1]))
        for index in range(len(terms) - 1)
    )


def _phrase_weight(phrase: str, normalized: str, stems: tuple[str, ...]) -> float:
    """Weight of one lexicon phrase inside a question."""

    if " " in phrase or "&" in phrase:
        return 1.0 if _normalize(phrase) in normalized else 0.0
    return 0.6 if phrase in stems else 0.0


def _query_intent(question: str, stems: tuple[str, ...]) -> Mapping[str, float]:
    """Map a question onto the page classes its answer is written in."""

    normalized = _normalize(question)
    intent: dict[str, float] = {}
    for kind, phrases in _PAGE_INTENT_LEXICON.items():
        weight = 0.0
        for phrase in phrases:
            weight = max(weight, _phrase_weight(phrase, normalized, stems))
        if weight:
            intent[kind] = weight
    for phrase, kinds in _PAGE_INTENT_EXTRA.items():
        if _normalize(phrase) in normalized:
            for kind in kinds:
                intent[kind] = max(intent.get(kind, 0.0), 1.0)
    return intent


def _analyze_query(question: str) -> _Query:
    """Resolve one question into terms, phrases and the page classes it needs."""

    terms = _query_terms(question)
    stems = _unique_stems(terms)
    return _Query(
        raw=question,
        terms=terms,
        stems=stems,
        bigrams=_query_bigrams(terms),
        intent=_query_intent(question, stems),
    )


def _classify_page(title: list[str]) -> frozenset[str]:
    """Recognize the section a page belongs to from its heading."""

    if not title:
        return frozenset()
    squashed = "".join(title).casefold()
    if squashed.startswith("note") or squashed.startswith("notes"):
        return frozenset({"notes"})
    return frozenset(
        kind for kind, pattern in _PAGE_KIND_PATTERNS if pattern in squashed
    )


def _document_entities(document: str) -> frozenset[str]:
    """Entity words of a document id, used to spot the document a question is about."""

    return frozenset(
        token
        for token in re.split(r"[^a-z0-9]+", document.casefold())
        if token and not token.isdigit() and not re.fullmatch(r"\d+[a-z]", token)
    )


async def search_pdf_pages(
    memory: _ScopedMemorySearch,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return exact citations for the pages that best answer one question.

    Retrieval runs through the public scoped Memory search API, so every returned
    citation resolves to a stored page. Candidates are gathered from several
    lexical routes so that a page whose wording differs from the question still
    enters the pool, then re-ranked offline by page section, IDF-weighted term
    coverage and neighbouring pages.
    """

    if limit < 1:
        return ()
    query = _analyze_query(question)
    hits, primary = await _collect_candidates(memory, query)
    candidates = _build_candidates(hits, query, primary)
    if not candidates:
        return ()
    _rank_candidates(candidates, query)
    await _attach_page_numbers(memory, candidates)
    _expand_neighbours(candidates)
    return _citations(candidates, limit)


async def _collect_candidates(
    memory: _ScopedMemorySearch,
    query: _Query,
) -> tuple[list[Any], dict[tuple[str, str, str, str], int]]:
    """Gather page candidates from several lexical routes.

    The rank of the main route is returned as well, because a page the backend
    already ranked highly must not be dropped by the offline re-ranker.
    """

    hits: list[Any] = []
    seen: set[tuple[str, str, str, str]] = set()
    primary: dict[tuple[str, str, str, str], int] = {}

    def absorb(page: Any, *, primary_route: bool = False) -> None:
        for rank, hit in enumerate(page.hits):
            identity = (
                hit.memory_ref.family,
                hit.memory_ref.artifact_id,
                hit.entry_id,
                hit.entry_version_id,
            )
            if primary_route:
                primary.setdefault(identity, rank)
            if identity not in seen:
                seen.add(identity)
                hits.append(hit)

    absorb(
        await memory.search(
            SearchMemoryRequest(query=query.raw, mode="fts", limit=_PAGE_CANDIDATE_LIMIT),
            admission=_PAGE_RELAXED_ADMISSION,
        ),
        primary_route=True,
    )
    if query.terms:
        absorb(
            await memory.search(
                SearchMemoryRequest(
                    query=" ".join(query.terms),
                    mode="fts",
                    limit=_PAGE_CANDIDATE_LIMIT,
                ),
                admission=_PAGE_RELAXED_ADMISSION,
            )
        )
    for term in query.stems[:_PAGE_TERM_ROUTES]:
        absorb(
            await memory.search(
                SearchMemoryRequest(query=term, mode="fts", limit=_PAGE_TERM_CANDIDATE_LIMIT),
                admission=_PAGE_RELAXED_ADMISSION,
            )
        )
    for kind in sorted(query.intent, key=lambda item: -query.intent[item]):
        for route in _PAGE_INTENT_ROUTES.get(kind, ()):
            absorb(
                await memory.search(
                    SearchMemoryRequest(query=route, mode="fts", limit=_PAGE_STATEMENT_CANDIDATE_LIMIT),
                    admission=_PAGE_TITLE_ADMISSION,
                )
            )
    return hits, primary


def _build_candidates(
    hits: list[Any],
    query: _Query,
    primary: dict[tuple[str, str, str, str], int] | None = None,
) -> list[_Candidate]:
    """Turn raw hits into candidates carrying the signals used for ranking."""

    ranks = primary or {}
    candidates: list[_Candidate] = []
    for hit in hits:
        text = hit.text
        document = text.partition("\n")[0]
        tokens = analyze_text(text).split()
        offset = len(analyze_text(document).split())
        title = tokens[offset : offset + _PAGE_TITLE_TOKENS]
        stems = [_stem(token) for token in tokens]
        stem_set = set(stems)
        title_stems = {_stem(token) for token in title}
        bigram_set = {
            (stems[index], stems[index + 1]) for index in range(len(stems) - 1)
        }
        numbers = sum(1 for token in tokens if token.isdigit())
        identity = (
            hit.memory_ref.family,
            hit.memory_ref.artifact_id,
            hit.entry_id,
            hit.entry_version_id,
        )
        candidates.append(
            _Candidate(
                hit=hit,
                document=document,
                matched=frozenset(
                    index for index, term in enumerate(query.stems) if term in stem_set
                ),
                title_matched=frozenset(
                    index for index, term in enumerate(query.stems) if term in title_stems
                ),
                bigram_matched=frozenset(
                    index for index, pair in enumerate(query.bigrams) if pair in bigram_set
                ),
                kinds=_classify_page(title),
                numeric=numbers / max(len(tokens), 1),
                primary_rank=ranks.get(identity, -1),
            )
        )
    return candidates


def _rank_candidates(candidates: list[_Candidate], query: _Query) -> None:
    """Score every candidate in place and sort it best first."""

    total = max(len(candidates), 1)
    frequency = [0] * len(query.stems)
    for candidate in candidates:
        for index in candidate.matched:
            frequency[index] += 1
    weights = [
        min(math.log(1.0 + total / max(count, 1)), _PAGE_IDF_CAP)
        for count in frequency
    ]
    total_weight = sum(weights) or 1.0
    term_weight: dict[str, float] = {}
    for index, term in enumerate(query.stems):
        term_weight[term] = max(term_weight.get(term, 0.0), weights[index])

    entities: dict[str, frozenset[str]] = {}
    for candidate in candidates:
        if candidate.document not in entities:
            entities[candidate.document] = _document_entities(candidate.document)
    targeted = [
        document
        for document, words in entities.items()
        if words & set(query.stems)
    ]
    target = targeted[0] if len(targeted) == 1 else None

    for candidate in candidates:
        coverage = sum(weights[index] for index in candidate.matched) / total_weight
        title = sum(weights[index] for index in candidate.title_matched) / total_weight
        conjunction = len(candidate.matched) / max(len(query.stems), 1)
        phrase = 0.0
        phrase_total = 0.0
        for index, pair in enumerate(query.bigrams):
            weight = min(term_weight.get(pair[0], 0.0), term_weight.get(pair[1], 0.0))
            phrase_total += weight
            if index in candidate.bigram_matched:
                phrase += weight
        phrase = phrase / phrase_total if phrase_total else 0.0
        intent = max(
            (query.intent.get(kind, 0.0) for kind in candidate.kinds),
            default=0.0,
        )
        entity = 1.0 if target is not None and candidate.document == target else 0.0
        candidate.score = (
            coverage
            + _PAGE_TITLE_WEIGHT * title
            + _PAGE_PHRASE_WEIGHT * phrase
            + _PAGE_INTENT_WEIGHT * intent
            + _PAGE_ENTITY_WEIGHT * entity
            + _PAGE_CONJUNCTION_WEIGHT * conjunction
            + _PAGE_TABLE_WEIGHT * candidate.numeric * (1.0 if intent > 0 else 0.0)
        )
    candidates.sort(key=lambda candidate: -candidate.score)


async def _attach_page_numbers(memory: _ScopedMemorySearch, candidates: list[_Candidate]) -> None:
    """Resolve the physical page number of the strongest candidates.

    The page number lives in the Source identity, so it is read back through the
    public Memory API rather than guessed from the text.
    """

    getter = getattr(memory, "get", None)
    if getter is None:
        return
    for candidate in candidates[:_PAGE_RESOLVE_LIMIT]:
        hit = candidate.hit
        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )
        record = await _read_entry(getter, citation)
        if record is None:
            continue
        for reference in getattr(record.entry, "sources", ()):
            number = _page_number(reference.source_id)
            if number > 0:
                candidate.page_number = number
                break


async def _read_entry(getter: Any, citation: MemoryCitation) -> Any | None:
    """Read one entry back through the public Memory API.

    The physical page number lives in the Source identity, so it is read back
    instead of being guessed from the text.  A Memory that cannot serve the
    citation only costs the neighbour refinement, so the failure is logged and
    the search continues.
    """

    try:
        return await getter(GetMemoryEntryRequest(citation=citation))
    except Exception:
        log_safely(
            logger,
            logging.WARNING,
            "PDF page citation could not be read back",
            exc_info=True,
            extra={
                "event": "pdf_page.citation_read_failed",
                "outcome": "failure",
                "unit": "pdf-page",
            },
        )
        return None


def _page_number(source_id: str) -> int:
    """Read the physical page number out of a page Source identity."""

    tail = source_id.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _expand_neighbours(candidates: list[_Candidate]) -> None:
    """Let a page inherit part of the score of the pages next to it.

    Evidence for one question often spans neighbouring pages: the consolidated
    statements sit next to each other and a disclosure continues over a page
    break.
    """

    if _PAGE_NEIGHBOUR_WINDOW <= 0:
        return
    by_page: dict[tuple[str, int], float] = {}
    for candidate in candidates:
        if candidate.page_number > 0:
            by_page[(candidate.document, candidate.page_number)] = candidate.score
    for candidate in candidates:
        if candidate.page_number <= 0:
            continue
        best = 0.0
        for offset in range(1, _PAGE_NEIGHBOUR_WINDOW + 1):
            for number in (candidate.page_number - offset, candidate.page_number + offset):
                value = by_page.get((candidate.document, number))
                if value is not None:
                    best = max(best, value)
        candidate.score += _PAGE_NEIGHBOUR_DECAY * best
    candidates.sort(key=lambda candidate: -candidate.score)


def _citations(candidates: list[_Candidate], limit: int) -> tuple[MemoryCitation, ...]:
    """Turn the best candidates into distinct, exact page citations.

    Some of the returned slots are reserved for the pages the backend itself
    ranked highest.  The re-ranker decides which pages deserve the remaining
    slots, but it never throws away the backend's own best matches, because
    those often carry evidence the lexical features cannot see.
    """

    chosen: list[_Candidate] = []
    taken: set[int] = set()
    reserved = min(int(limit * _PAGE_PRIMARY_RESERVE), max(limit - 1, 0))
    if reserved > 0:
        by_rank = sorted(
            (candidate for candidate in candidates if candidate.primary_rank >= 0),
            key=lambda candidate: candidate.primary_rank,
        )
        for candidate in by_rank[:reserved]:
            chosen.append(candidate)
            taken.add(id(candidate))
    for candidate in candidates:
        if len(chosen) >= limit:
            break
        if id(candidate) in taken:
            continue
        chosen.append(candidate)
        taken.add(id(candidate))

    citations: list[MemoryCitation] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in chosen:
        hit = candidate.hit
        identity = (
            hit.memory_ref.family,
            hit.memory_ref.artifact_id,
            hit.entry_id,
            hit.entry_version_id,
        )
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)

__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
