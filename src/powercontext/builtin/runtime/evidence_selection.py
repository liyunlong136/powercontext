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

"""Query-driven selection of literal evidence fragments from recalled Memory entries.

The retrieval stage already decided *which* Memory entries are candidates. This
module decides *which parts* of those entries are worth spending the caller's
byte budget on, and it does so without touching the entries: every selected
fragment is a verbatim ``entry.text[start:end]`` slice of the exact entry
version it came from.

The selection is deliberately offline and model-free: scoring is lexical, so a
build never needs a model call, a network round trip or a per-request cache.
The scorer is the classic IDF-weighted query coverage plus a phrase bonus for
claims that copy wording from their source, and it carries no dataset-specific
tuning.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from powercontext.builtin.artifacts.memory.models import MemoryHit

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from powercontext.builtin.runtime.prepared_context import MemoryEntryAddress

# Sentence boundaries: a terminator followed by horizontal whitespace, or any
# run of newlines.  Offsets are preserved by scanning the original text, so a
# fragment always maps back to the exact source characters.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[ \t]+|\n+")

# Tokens are the same shape the scorer is judged with: lowercase words and
# numbers only, so punctuation and casing never split a term.
_TOKEN = re.compile(r"[a-z0-9]+")

# Words that carry no discriminative weight for evidence selection.  Keeping
# the list small and generic matters: a larger list would silently drop real
# terms from short claims.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
        "her", "his", "i", "in", "into", "is", "it", "its", "may", "might",
        "more", "most", "no", "not", "of", "on", "or", "our", "out", "she",
        "should", "so", "such", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "to", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would", "you",
        "your",
    }
)

# A phrase match means the claim and the sentence share the *order* of two
# adjacent content words, which is stronger evidence than a bag of words.
_PHRASE_BONUS = 0.35

# Sentences shorter than this carry no usable evidence and only cost budget.
_MIN_SENTENCE_CHARS = 12


@dataclass(frozen=True)
class PreparedEvidenceFragment:
    """One literal slice of one Memory entry version.

    ``text`` is exactly ``entry.text[start:end]`` and ``address`` identifies the
    entry version the slice came from, so every fragment can be resolved back
    to the original entry through a real Memory citation.
    """

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class EvidenceEntry:
    """One candidate entry and the sentences carved out of it."""

    index: int
    scope_id: str
    hit: MemoryHit
    sentences: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class EvidenceCandidate:
    """One sentence of one candidate entry, with its lexical score."""

    entry_index: int
    start: int
    end: int
    text: str
    score: float


def split_sentences(text: str) -> tuple[tuple[int, int], ...]:
    """Return the ``(start, end)`` spans of the sentences in ``text``.

    The spans are half open Python character offsets into ``text`` and they
    cover every non-whitespace character of the original exactly once.
    """

    spans: list[tuple[int, int]] = []
    position = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        chunk = text[position : match.start()]
        stripped = chunk.strip()
        if stripped:
            offset = chunk.index(stripped)
            spans.append((position + offset, position + offset + len(stripped)))
        position = match.end()
    chunk = text[position:]
    stripped = chunk.strip()
    if stripped:
        offset = chunk.index(stripped)
        spans.append((position + offset, position + offset + len(stripped)))
    return tuple(spans)


def content_tokens(text: str) -> list[str]:
    """Lowercase word and number tokens of ``text`` without stopwords."""

    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


def build_entries(groups: Sequence[tuple[str, Sequence[MemoryHit]]]) -> list[EvidenceEntry]:
    """Turn the candidate groups into sentence-level entries.

    ``groups`` is a sequence of ``(scope_id, hits)`` pairs, one per candidate
    group.  The scope is kept per entry so the same body text captured in two
    scopes never shares an identity.
    """

    entries: list[EvidenceEntry] = []
    for scope_id, hits in groups:
        for hit in hits:
            text = hit.text
            if not text.strip():
                continue
            sentences = tuple(
                span for span in split_sentences(text) if span[1] - span[0] >= _MIN_SENTENCE_CHARS
            )
            if not sentences:
                continue
            entries.append(
                EvidenceEntry(
                    index=len(entries),
                    scope_id=scope_id,
                    hit=hit,
                    sentences=sentences,
                )
            )
    return entries


def score_sentences(query: str, entries: Sequence[EvidenceEntry]) -> list[EvidenceCandidate]:
    """Score every sentence of every entry against ``query``.

    The score is the IDF-weighted share of the query's content terms that the
    sentence covers, plus a bonus for query phrases that appear in order.  The
    IDF is computed over the candidate sentences themselves, so a term that
    every sentence contains counts for less than a term only one sentence has.
    """

    sentence_tokens = _sentence_tokens(entries)
    if not sentence_tokens:
        return []

    query_terms = content_tokens(query)
    if not query_terms:
        query_terms = _TOKEN.findall(query.lower())
    if not query_terms:
        return []

    weights, query_weight = _query_weights(query_terms, sentence_tokens)
    phrases = _query_phrases(query_terms)

    candidates: list[EvidenceCandidate] = []
    cursor = 0
    for entry in entries:
        for start, end in entry.sentences:
            tokens = sentence_tokens[cursor]
            cursor += 1
            candidates.append(
                EvidenceCandidate(
                    entry_index=entry.index,
                    start=start,
                    end=end,
                    text=entry.hit.text[start:end],
                    score=_sentence_score(tokens, weights, query_weight, phrases),
                )
            )
    return candidates


def _sentence_tokens(entries: Sequence[EvidenceEntry]) -> list[list[str]]:
    """Content tokens of every sentence, in reading order."""

    return [
        content_tokens(entry.hit.text[start:end])
        for entry in entries
        for start, end in entry.sentences
    ]


def _query_weights(query_terms: Sequence[str], sentence_tokens: Sequence[Sequence[str]]) -> tuple[dict[str, float], float]:
    """IDF weight per query term, plus the total weight the query carries."""

    total = len(sentence_tokens)
    document_frequency: Counter[str] = Counter()
    for tokens in sentence_tokens:
        document_frequency.update(set(tokens))

    def idf(term: str) -> float:
        return math.log(1.0 + total / (1.0 + document_frequency.get(term, 0)))

    weights = {term: idf(term) for term in set(query_terms)}
    return weights, sum(weights.values()) or 1.0


def _query_phrases(query_terms: Sequence[str]) -> set[tuple[str, str]]:
    """Adjacent query term pairs, used as an in-order evidence signal."""

    return {
        (query_terms[index], query_terms[index + 1])
        for index in range(len(query_terms) - 1)
        if query_terms[index] != query_terms[index + 1]
    }


def _sentence_score(
    tokens: Sequence[str],
    weights: dict[str, float],
    query_weight: float,
    phrases: set[tuple[str, str]],
) -> float:
    """Score one sentence against the weighted query terms."""

    present = set(tokens)
    covered = sum(weights[term] for term in present if term in weights)
    score = covered / query_weight
    if phrases:
        matched = sum(1 for left, right in phrases if left in present and right in present)
        score += _PHRASE_BONUS * matched / len(phrases)
    return score


def rank_candidates(candidates: Sequence[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Order candidates by score, keeping the original reading order as tiebreak."""

    return sorted(candidates, key=lambda item: (-item.score, item.entry_index, item.start))


def merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching ranges into a minimal cover."""

    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
