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

"""Coalesced embedding calls with a bounded vector cache.

Memory writes and vector searches embed overlapping texts, and several scopes
often submit the same batch at the same time. This wrapper keeps one shared
view of that work: identical texts are embedded once per underlying request,
concurrent callers share the request that is still running, and successful
vectors are reused from a bounded LRU cache.

The wrapper never changes the shape of the underlying model contract. It
exposes the wrapped model profile, returns the existing
:class:`~powercontext.builtin.inference.models.EmbeddingResult`, and reports
the usage of the distinct underlying batches each caller actually waited for.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Final, cast

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.protocols import EmbeddingModel

DEFAULT_MAX_BATCH_SIZE: Final[int] = 64
DEFAULT_COALESCE_MS: Final[float] = 5.0
DEFAULT_CACHE_SIZE: Final[int] = 1024


class _Batch:
    """One underlying request, shared by every caller waiting for its texts."""

    __slots__ = ("texts", "usage")

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.usage: InferenceUsage | None = None


class _Entry:
    """One text tracked from registration until its vector is delivered."""

    __slots__ = ("batch", "future", "text", "vector")

    def __init__(self, text: str, future: asyncio.Future[EmbeddingVector] | None = None) -> None:
        self.text = text
        self.vector: EmbeddingVector | None = None
        self.future = future
        self.batch: _Batch | None = None


class CoalescingEmbeddingModel:
    """Share embedding work across calls and reuse successful vectors.

    The wrapper deduplicates the texts of one call, merges the texts of calls
    that arrive inside the coalescing window, and splits the merged texts into
    underlying requests of at most ``max_batch_size`` deduplicated texts.
    A text that is already being embedded is never embedded twice: late
    callers join the running request instead of starting a new one.

    Cancelling one caller never cancels the underlying request other callers
    wait for. Failed requests are not cached, so a later call can retry them.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Wrap one embedding model with coalescing and a bounded vector cache."""

        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self.profile: EmbeddingProfile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_ms = coalesce_ms
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, _Entry] = {}
        self._pending: list[str] = []
        self._leading = False
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def model(self) -> EmbeddingModel:
        """Return the wrapped embedding model."""

        return self._model

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of deduplicated texts per underlying request."""

        return self._max_batch_size

    @property
    def coalesce_ms(self) -> float:
        """Return the coalescing window in milliseconds."""

        return self._coalesce_ms

    @property
    def cache_size(self) -> int:
        """Return the capacity of the successful-vector LRU cache."""

        return self._cache_size

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
        """Embed one ordered text batch, sharing work and vectors across calls."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        entries = [self._register(text) for text in texts]
        if not self._leading and self._pending:
            await self._lead()
        await self._await_entries(entries)
        return self._result(entries)

    def _register(self, text: str) -> _Entry:
        """Resolve one text to a cached vector or to a shared pending request."""

        cached = self._cached_vector(text)
        if cached is not None:
            entry = _Entry(text)
            entry.vector = cached
            return entry
        entry = self._inflight.get(text)
        if entry is None:
            future: asyncio.Future[EmbeddingVector] = asyncio.get_running_loop().create_future()
            entry = _Entry(text, future)
            future.add_done_callback(_consume_failure)
            self._inflight[text] = entry
            self._pending.append(text)
        return entry

    async def _lead(self) -> None:
        """Own the coalescing window and dispatch the texts waiting in it."""

        self._leading = True
        try:
            while True:
                await self._wait_window()
                self._dispatch_pending()
                if not self._pending:
                    break
        finally:
            self._leading = False
            # A cancelled leader still owes its waiters the texts it collected.
            self._dispatch_pending()

    async def _wait_window(self) -> None:
        """Wait one coalescing window so concurrent calls can merge."""

        delay = self._coalesce_ms / 1000
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)

    def _dispatch_pending(self) -> None:
        """Split the pending texts into bounded underlying requests."""

        pending = self._pending
        if not pending:
            return
        self._pending = []
        for start in range(0, len(pending), self._max_batch_size):
            batch = tuple(pending[start : start + self._max_batch_size])
            record = _Batch(batch)
            for text in batch:
                self._inflight[text].batch = record
            task = asyncio.create_task(self._run_batch(record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_batch(self, record: _Batch) -> None:
        """Run one underlying request and deliver its vectors to every waiter."""

        entries = [self._inflight[text] for text in record.texts]
        try:
            result = await self._model.embed(record.texts)
        except asyncio.CancelledError as error:
            self._fail(entries, error)
            raise
        except Exception as error:
            self._fail(entries, error)
            return
        except BaseException as error:
            # A provider that aborts the interpreter must not strand the waiters.
            self._fail(entries, error)
            raise
        if len(result.vectors) != len(record.texts):
            self._fail(
                entries,
                ValueError(
                    f"embedding provider returned {len(result.vectors)} vectors "
                    f"for {len(record.texts)} texts"
                ),
            )
            return
        record.usage = result.usage
        for entry, vector in zip(entries, result.vectors, strict=True):
            self._inflight.pop(entry.text, None)
            self._store(entry.text, vector)
            entry.vector = vector
            if entry.future is not None and not entry.future.done():
                entry.future.set_result(vector)

    def _fail(self, entries: list[_Entry], error: BaseException) -> None:
        """Deliver one failure to the waiters of a batch without caching it."""

        for entry in entries:
            self._inflight.pop(entry.text, None)
            if entry.future is not None and not entry.future.done():
                entry.future.set_exception(error)

    async def _await_entries(self, entries: list[_Entry]) -> None:
        """Wait for every requested text, isolating the caller from cancellation."""

        futures = [entry.future for entry in entries if entry.future is not None]
        if not futures:
            return
        results = await asyncio.gather(
            *(asyncio.shield(future) for future in futures),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _result(self, entries: list[_Entry]) -> EmbeddingResult:
        """Rebuild the caller result in its original order, duplicates included."""

        vectors = tuple(cast(EmbeddingVector, entry.vector) for entry in entries)
        batches = {entry.batch for entry in entries if entry.batch is not None}
        return EmbeddingResult(vectors=vectors, usage=self._usage(batches))

    def _usage(self, batches: set[_Batch]) -> InferenceUsage:
        """Sum the usage of the distinct batches one caller waited for."""

        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        for batch in batches:
            usage = batch.usage
            if usage is None:
                continue
            requests += usage.requests
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens
        return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)

    def _cached_vector(self, text: str) -> EmbeddingVector | None:
        """Return one cached vector, refreshing its recent-use order."""

        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _store(self, text: str, vector: EmbeddingVector) -> None:
        """Cache one successful vector, evicting the least recently used entry."""

        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _consume_failure(future: asyncio.Future[EmbeddingVector]) -> None:
    """Retrieve a batch failure so an abandoned waiter never warns."""

    if not future.cancelled():
        future.exception()


__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COALESCE_MS",
    "DEFAULT_MAX_BATCH_SIZE",
    "CoalescingEmbeddingModel",
]
