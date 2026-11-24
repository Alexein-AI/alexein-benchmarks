"""Alexein, through its public Python client (`pip install alexein`), as any app would use it.

Each unit of a run is an app user of its own, `bench-<run>-<digest of bench and unit>` (an app
user id is at most 35 characters): its items are added
in time order, a question is asked only once everything before its date is processed, and the
user is erased when the unit is done.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypeVar, cast

from alexein import AlexeinError, APIConnectionError, InvalidRequestError, Memory, NotFoundError
from alexein._calls import Call

T = TypeVar("T")

# One add carries up to 500 items and a megabyte; status asks after as many items.
_BATCH = 500
_BYTES = 900_000
_POLL = 2.0
# What a wait waits for: searchable, or with what it says learned too.
STORED = frozenset({"stored", "processed"})
PROCESSED = frozenset({"processed"})
# How often a wait says how far it has got: at every look.
_REPORT = 0.0
_PREFIX = "bench-"


# An outage - a 429, a 5xx, no connection - is waited out, doubling from 2s, for about five
# minutes in all; a long run should not lose questions to a blip.
_OUTAGE_TRIES = 8
_OUTAGE_WAIT = 2.0


class MemoryTimeoutError(RuntimeError):
    pass


def outage(error: Exception) -> bool:
    return isinstance(error, APIConnectionError) or (
        isinstance(error, AlexeinError) and error.outage
    )


def short(error: Exception) -> str:
    """An error in a line: a gateway's HTML page is not one."""
    if isinstance(error, AlexeinError) and error.status is not None:
        detail = (error.detail or "").strip()
        detail = "" if detail.startswith("<") else f": {detail[:200]}"
        return f"Alexein answered {error.status}{detail}"
    return f"{type(error).__name__}: {str(error)[:200]}"


def patient(call: Callable[[], T]) -> T:
    """`call`, again after an outage, until it answers or about five minutes have gone."""
    for attempt in range(_OUTAGE_TRIES):
        try:
            return call()
        except Exception as error:
            if not outage(error) or attempt == _OUTAGE_TRIES - 1:
                raise
            time.sleep(_OUTAGE_WAIT * 2**attempt)
    raise AssertionError("unreachable")


def user_id(bench: str, unit: str, run: str, *, variant: str) -> str:
    """`bench-20261117a1b2c3-806f0f51c1`: the run, then which unit of which benchmark and
    variant. Variants of one run never share a memory: RHELM's `with` holds emails and
    attachments that `without` must not see."""
    digest = hashlib.sha256(f"{bench}/{variant}/{unit}".encode()).hexdigest()[:10]
    return f"{_PREFIX}{run}-{digest}"


def batches(items: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Items in order, cut so that no add is over 500 items or near a megabyte."""
    out: list[list[dict[str, Any]]] = [[]]
    size = 0
    for item in items:
        weight = len(json.dumps(item, ensure_ascii=False).encode())
        if out[-1] and (len(out[-1]) >= _BATCH or size + weight > _BYTES):
            out.append([])
            size = 0
        out[-1].append(item)
        size += weight
    return [b for b in out if b]


class Account:
    """One app user's memory for the length of one unit."""

    def __init__(self, memory: Memory, user: str, *, timeout: float) -> None:
        self.user = memory.user(user)
        self.name = user
        self.timeout = timeout
        self.added = 0
        self.searches = 0
        # Whether the API takes `/v2/memory/settle`; learned at the first try.
        self.settles: bool | None = None

    def add(self, items: Sequence[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for batch in batches(items):
            ids += patient(partial(self.user.add, cast("list[Any]", batch))).ids
            self.added += len(batch)
        return ids

    def held(self, ids: Sequence[str]) -> set[str]:
        """Which of these items memory already has, processed or on its way."""
        out: set[str] = set()
        for start in range(0, len(ids), _BATCH):
            chunk = list(ids[start : start + _BATCH])
            try:
                status = patient(partial(self.user.status, chunk)).by_id
            except NotFoundError:  # an app user with no memory yet holds nothing
                return set()
            out |= {i for i in chunk if status.get(i, "unknown") != "unknown"}
        return out

    def settle(self) -> bool:
        """Tell memory that what has been added is complete, so it is processed now rather than
        after a quiet spell (`POST /v2/memory/settle`). False where the API does not have it:
        then its items are processed at memory's own pace, and waited for."""
        if self.settles is False:
            return False
        # Through the client's own transport: the published client has no `settle()` yet.
        call = Call("POST", "/v2/memory/settle", lambda raw: raw, {})
        try:
            patient(partial(self.user._send, call))
        except (NotFoundError, InvalidRequestError) as error:
            if error.status not in (404, 405) or "no memory" in (error.detail or ""):
                raise
            self.settles = False
            return False
        self.settles = True
        return True

    def wait(
        self,
        ids: Sequence[str],
        stop: threading.Event | None = None,
        report: Callable[[int, int, float], None] | None = None,
        until: frozenset[str] = PROCESSED,
    ) -> float:
        """Block until every item has reached one of `until` - processed by default - or `stop`
        is set. Seconds waited. `report` is told (reached, of, seconds) while it waits."""
        started, pending = time.monotonic(), list(ids)
        reported = started
        while pending and not (stop and stop.is_set()):
            still: list[str] = []
            for start in range(0, len(pending), _BATCH):
                chunk = pending[start : start + _BATCH]
                status = patient(partial(self.user.status, chunk)).by_id
                still += [i for i in chunk if status.get(i) not in until]
                lost = [i for i in chunk if status.get(i) == "unknown"]
                if lost:
                    raise MemoryTimeoutError(
                        f"{self.name}: {len(lost)} items unknown, e.g. {lost[0]}"
                    )
            pending = still
            now = time.monotonic()
            if pending and report is not None and now - reported >= _REPORT:
                report(len(ids) - len(pending), len(ids), now - started)
                reported = now
            if pending and now - started > self.timeout:
                raise MemoryTimeoutError(
                    f"{self.name}: {len(pending)} items not processed after {self.timeout:.0f}s"
                )
            if pending and stop is not None:
                stop.wait(_POLL)
            elif pending:
                time.sleep(_POLL)
        return time.monotonic() - started

    def search(
        self, query: str, max_chars: int, before: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """The context a model reads, as the client renders it, and the raw answer. `before`
        is the day the question is asked: memory reads its time words from that day."""
        self.searches += 1
        found = patient(partial(self.user.search, query, before=before))
        return found.context(max_chars=max_chars), found.raw

    def delete(self) -> None:
        patient(self.user.delete)


def leftovers(memory: Memory, run: str | None = None) -> list[str]:
    """App users a run made and did not erase: every `bench-` user, or one run's."""
    return [
        u.user
        for u in memory.iter_users()
        if u.user.startswith(_PREFIX + (f"{run}-" if run else ""))
    ]
