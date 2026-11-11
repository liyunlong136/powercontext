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

"""HTTP rerank-service adapter for coarse Memory search candidates.

The adapter speaks one fixed offline protocol against a caller-supplied
``httpx.AsyncClient``:

- **request** — one ``POST`` per batch of at most ``batch_size`` candidates with
  the JSON body ``{"query": ..., "candidates": [...]}``, where every candidate
  carries its one-based ``rank`` inside the **complete** candidate pool and the
  candidate ``text``;
- **response** — one JSON body that contains exactly ``{"scores": [...]}``, where
  every record carries exactly ``{"rank": ..., "score": ...}``.

Ranks are never renumbered per batch: the service must score the original
candidate, so every rank is sent exactly once and the final selection is taken
over the union of all batch results. A late batch can therefore still contribute
the top-ranked candidate.
"""

from __future__ import annotations

import json
import math
from typing import Final

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

HTTP_RERANKER_POLICY_ID: Final[str] = "http-cross-encoder-v1"
HTTP_RERANKER_BATCH_SIZE: Final[int] = 32

_RESPONSE_FIELDS: Final[frozenset[str]] = frozenset({"scores"})
_SCORE_FIELDS: Final[frozenset[str]] = frozenset({"rank", "score"})


class HTTPMemoryReranker:
    """Select final Memory hits with the scores of one HTTP rerank service.

    The adapter only reorders candidates: it never rewrites candidate text and
    never supplies entry identities, so the Runtime still resolves every
    selected rank to the original ``MemoryHit`` and keeps its exact Artifact,
    entry, and Revision identity.

    Every candidate is scored exactly once and the global selection is taken
    over the union of all batch results, so a high-scoring candidate from a
    late batch still reaches the caller. Ties are broken by the original global
    rank, which keeps the selection stable and independent of the batch layout.
    """

    policy_id: str

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = HTTP_RERANKER_BATCH_SIZE,
        policy_id: str = HTTP_RERANKER_POLICY_ID,
    ) -> None:
        """Bind one rerank endpoint to one caller-owned HTTP client."""

        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty string")  # noqa: TRY003
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("rerank policy_id must be a non-empty string")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
    ) -> MemoryRerankDecision:
        """Score every coarse candidate and return the globally best ``limit`` ranks."""

        pool = tuple(candidates)
        if not isinstance(query, str) or not query:
            raise ValueError("rerank query must be a non-empty string")  # noqa: TRY003
        if not pool:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > len(pool):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003
        for candidate in pool:
            if not isinstance(candidate.text, str) or not candidate.text:
                raise ValueError("rerank candidates must carry non-empty text")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(pool), self._batch_size):
            batch = pool[start : start + self._batch_size]
            payload = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidate.text}
                    for rank, candidate in enumerate(batch, start=start + 1)
                ],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            scores.update(_validated_scores(_decoded_payload(response), expected=_batch_ranks(batch, start=start + 1)))

        selected = tuple(rank for rank, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=0,
            used_fallback=False,
        )


def _batch_ranks(batch: tuple[MemoryHit, ...], *, start: int) -> frozenset[int]:
    """Return the global ranks one batch is allowed to score."""

    return frozenset(range(start, start + len(batch)))


def _decoded_payload(response: httpx.Response) -> object:
    """Return the decoded JSON body of one successful rerank response."""

    if not response.is_success:
        response.raise_for_status()
        raise httpx.HTTPStatusError(  # noqa: TRY003
            f"rerank service returned HTTP {response.status_code}",
            request=response.request,
            response=response,
        )
    try:
        text = response.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("rerank response is not valid UTF-8") from error  # noqa: TRY003
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"rerank response is not valid JSON: {error}") from error  # noqa: TRY003


def _validated_scores(payload: object, *, expected: frozenset[int]) -> dict[int, float]:
    """Return the scores one batch reported, rejecting every invalid coverage."""

    if not isinstance(payload, dict):
        raise ValueError("rerank response must be a JSON object")  # noqa: TRY003, TRY004
    if set(payload) != _RESPONSE_FIELDS:
        raise ValueError("rerank response must contain exactly the 'scores' field")  # noqa: TRY003
    records = payload["scores"]
    if not isinstance(records, list):
        raise ValueError("rerank response field 'scores' must be a list")  # noqa: TRY003, TRY004
    scores: dict[int, float] = {}
    for record in records:
        rank, score = _score_record(record)
        if rank not in expected:
            raise ValueError(f"rerank response scored rank {rank}, which is not part of the requested batch")  # noqa: TRY003
        if rank in scores:
            raise ValueError(f"rerank response scored rank {rank} more than once")  # noqa: TRY003
        scores[rank] = score
    if set(scores) != expected:
        raise ValueError("rerank response must score every requested rank exactly once")  # noqa: TRY003
    return scores


def _score_record(record: object) -> tuple[int, float]:
    """Return the validated rank and finite score of one response record."""

    if not isinstance(record, dict):
        raise ValueError("rerank response score records must be JSON objects")  # noqa: TRY003, TRY004
    if set(record) != _SCORE_FIELDS:
        raise ValueError("rerank response score records must contain exactly the 'rank' and 'score' fields")  # noqa: TRY003
    rank = record["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("rerank response score field 'rank' must be an integer")  # noqa: TRY003, TRY004
    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("rerank response score field 'score' must be a number")  # noqa: TRY003, TRY004
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("rerank response score field 'score' must be a finite number")  # noqa: TRY003
    return rank, value


__all__ = [
    "HTTP_RERANKER_BATCH_SIZE",
    "HTTP_RERANKER_POLICY_ID",
    "HTTPMemoryReranker",
]
