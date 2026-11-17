"""A run: every unit of a benchmark through Alexein, its questions answered and scored.

Per unit: an app user of its own; the unit's items added in time order, a question date at a
time, each step waited on until memory has processed it; each question searched, answered from
what the search returned, and scored; the user erased once every question is answered.

What a run leaves in its folder, each written as it goes:

- `results.jsonl`: one row per question as it is scored - the question, the expected answer,
  the model's, the score. A row is on disk the moment it is written.
- `traces.jsonl`: per question, everything behind its row: what was searched, the search's whole
  answer, the context the model read, and every model call's request and reply.
- `running.txt` and `summary.json`: the score so far, every `report_every` questions.

A stopped run carries on with `--resume`: questions already scored are not asked again, and a
unit that was part way through keeps its app user and gets only the items it does not hold yet.
If its memory holds anything a question still to be asked must not see, it is built again from
nothing instead.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexein import Memory

from alexein_bench.core import Bench, Json, Question, Unit
from alexein_bench.memory import STORED, Account, short, user_id
from alexein_bench.models import Models

Step = tuple[list[Json], list[Question]]


class StoppedError(Exception):
    """A unit left part way when the run was stopped; its app user is kept for --resume."""


@dataclass(frozen=True)
class Settings:
    bench: Bench
    variant: str
    run: str
    out: Path
    workers: int = 4
    wait: float = 3600.0
    keep: bool = False
    resume: bool = False
    report_every: int = 10
    # Seconds between status lines: a run is never quiet for longer, whatever it is doing.
    heartbeat: float = 30.0


def steps(unit: Unit) -> list[Step]:
    """The unit as (items to add, questions to ask then), in time order: each question waits
    for every item dated before its cutoff, and no item from on or after it. Items after the
    last question are added only if a question is asked after everything."""
    items = sorted(unit.items, key=lambda i: str(i["at"]))
    timed = sorted((q for q in unit.questions if q.cutoff is not None), key=lambda q: str(q.cutoff))
    untimed = [q for q in unit.questions if q.cutoff is None]
    out: list[Step] = []
    taken = 0
    for question in timed:
        cut = str(question.cutoff)
        upto = taken
        while upto < len(items) and str(items[upto]["at"]) < cut:
            upto += 1
        if out and upto == taken:
            out[-1][1].append(question)
        else:
            out.append((items[taken:upto], [question]))
        taken = upto
    if untimed:
        out.append((items[taken:], untimed))
    return out


def trimmed(units: Iterable[Unit], limit: int | None, unit_limit: int | None) -> list[Unit]:
    """At most `unit_limit` units and `limit` questions, whole units first."""
    kept: list[Unit] = []
    left = limit
    for unit in units:
        if unit_limit is not None and len(kept) >= unit_limit:
            break
        if left is not None and left <= 0:
            break
        questions = unit.questions if left is None else unit.questions[:left]
        if left is not None:
            left -= len(questions)
        kept.append(Unit(unit.id, unit.items, list(questions)))
    return kept


def plan_resume(plan: list[Step], held: set[str], done: set[str]) -> int | None:
    """Where a unit picks up: the first step with a question still to ask, if what memory
    holds is all from that step or before it; None when it holds something later, and must be
    built again from nothing."""
    first = next(
        (n for n, (_, qs) in enumerate(plan) if any(q.id not in done for q in qs)), len(plan)
    )
    later = {str(i["id"]) for items, _ in plan[first + 1 :] for i in items}
    return None if held & later else first


def _clock(seconds: float) -> str:
    """`42s`, `7m05s`, `2h03m`."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02}s"
    return f"{s // 3600}h{s % 3600 // 60:02}m"


def _read(path: Path) -> list[Json]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        # A line cut short by a crash mid-write is the one question to ask again.
        with contextlib.suppress(json.JSONDecodeError):
            rows.append(json.loads(line))
    return rows


class Runner:
    def __init__(self, settings: Settings, memory: Memory, models: Models) -> None:
        self.s = settings
        self.memory = memory
        self.models = models
        self.path = settings.out / "results.jsonl"
        self.traces = settings.out / "traces.jsonl"
        self._lock = threading.Lock()
        self._rows: dict[str, Json] = {r["id"]: r for r in _read(self.path)}
        self.done = {i for i, r in self._rows.items() if "error" not in r}
        self.added = 0
        self.searches = 0
        self._since_report = 0
        self._progress: Callable[[str], None] = lambda _line: None
        self._stop = threading.Event()
        # What each unit is doing now, and since when: the status line reads it.
        self._state: dict[str, tuple[str, float]] = {}
        self._finished = 0
        self._failed = 0
        self._total = 0
        self._started = time.monotonic()
        self._units = 0

    def rows(self) -> list[Json]:
        with self._lock:
            return list(self._rows.values())

    # --- writing as it goes

    def _append(self, path: Path, record: Json) -> None:
        with path.open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write(self, row: Json, trace: Json | None = None) -> None:
        with self._lock:
            if trace is not None:
                self._append(self.traces, trace)
            self._append(self.path, row)
            self._rows[row["id"]] = row
            self._since_report += 1
            due = self._since_report >= self.s.report_every
            if due:
                self._since_report = 0
        if due:
            self.report()

    def summary(self) -> dict[str, Any]:
        rows = self.rows()
        out = summarize(self.s.bench, self.s.variant, rows)
        with self._lock:
            added, searches = self.added, self.searches
        return out | {
            "bench": self.s.bench.name,
            "variant": self.s.variant,
            "run": self.s.run,
            "models": self.models.bill.as_dict(),
            "alexein": {"items_added": added, "searches": searches},
        }

    def report(self) -> dict[str, Any]:
        """The score so far: printed, appended to `running.txt`, and `summary.json` rewritten,
        so a run stopped at any point has a current summary."""
        summary = self.summary()
        scores = summary["scores"]
        head = scores.get(self.s.variant, "-")
        parts = [f"{k.split(':', 1)[1]} {v}" for k, v in scores.items() if ":" in k]
        line = (
            f"{summary['questions']} answered · {self.s.variant} {head}"
            + (f" · {' · '.join(parts)}" if parts else "")
            + f" · errors {summary['errors']} · models ${summary['models']['usd']:.2f}"
        )
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            with (self.s.out / "running.txt").open("a") as out:
                out.write(f"[{stamp}] {line}\n")
            (self.s.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self._progress(line)
        return summary

    # --- the run

    # --- the status line

    def _set(self, unit: str, phase: str | None) -> None:
        with self._lock:
            if phase is None:
                self._state.pop(unit, None)
            elif self._state.get(unit, ("", 0.0))[0].split(" ")[0] == phase.split(" ")[0]:
                self._state[unit] = (phase, self._state[unit][1])
            else:
                self._state[unit] = (phase, time.monotonic())

    def status(self) -> str:
        """One line: questions answered, units done and running, what each running one is
        doing and for how long, and the spend so far."""
        now = time.monotonic()
        with self._lock:
            answered = sum(1 for r in self._rows.values() if "error" not in r)
            errors = len(self._rows) - answered
            running = sorted(self._state.items())
            finished, failed = self._finished, self._failed
        waiting = self._units - finished - failed - len(running)
        parts = [
            f"{_clock(now - self._started)}",
            f"{answered:,}/{self._total:,} answered" + (f", {errors} errors" if errors else ""),
            f"units {finished} done, {len(running)} running, {max(waiting, 0)} waiting"
            + (f", {failed} stopped" if failed else ""),
            *(f"{unit} {phase} ({_clock(now - since)})" for unit, (phase, since) in running),
            f"models ${self.models.bill.usd:.2f}",
        ]
        return " · ".join(parts)

    def _beat(self) -> None:
        while not self._stop.wait(self.s.heartbeat):
            self._progress(self.status())

    # --- the run

    def run(self, units: list[Unit], progress: Callable[[str], None]) -> dict[str, Any]:
        self._progress = progress
        todo = [u for u in units if any(q.id not in self.done for q in u.questions)]
        self._units = len(todo)
        self._total = sum(len(u.questions) for u in units)
        self._started = time.monotonic()
        progress(f"{len(todo)} of {len(units)} units to run, {self.s.workers} at a time")
        beat = threading.Thread(target=self._beat, daemon=True)
        beat.start()
        pool = ThreadPoolExecutor(max_workers=self.s.workers)
        try:
            futures = {pool.submit(self._unit, u, progress): u for u in todo}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    future.result()
                except StoppedError:
                    pass
                except Exception as error:  # one unit failing leaves the others running
                    with self._lock:
                        self._failed += 1
                    progress(
                        f"{unit.id}: stopped - {short(error)}. Its app user is kept for --resume."
                    )
                    for q in unit.questions:
                        if q.id not in self.done and q.id not in self._rows:
                            self._write({"id": q.id, "unit": unit.id, "error": short(error)})
        except BaseException:
            # Ctrl-C: the questions being asked finish and are written; nothing new starts.
            self._stop.set()
            progress("stopping: finishing the questions in flight")
            pool.shutdown(wait=True, cancel_futures=True)
            self.report()
            raise
        pool.shutdown(wait=True)
        self._stop.set()  # ends the status line
        return self.report()

    def _start(self, account: Account, plan: list[Step]) -> tuple[int, set[str]]:
        """Which step to begin at, and the items memory already holds."""
        if not self.s.resume:
            return 0, set()
        ids = [str(i["id"]) for items, _ in plan for i in items]
        held = account.held(ids)
        first = plan_resume(plan, held, self.done)
        if first is None:
            account.delete()
            return 0, set()
        return first, held

    def _unit(self, unit: Unit, progress: Callable[[str], None]) -> None:
        name = user_id(self.s.bench.name, unit.id, self.s.run)
        account = Account(self.memory, name, timeout=self.s.wait)
        plan = steps(unit)
        started, asked = time.monotonic(), 0
        mine = [q for _, qs in plan for q in qs if q.id not in self.done]
        try:
            self._set(unit.id, "starting")
            first, held = self._start(account, plan)
            if held:
                progress(f"{unit.id}: resuming, {len(held)} items already in memory")
            # Everything before the first step to ask is added at once: none of its
            # questions is left to ask in between.
            before = [i for items, _ in plan[:first] for i in items]
            pending = [(before + plan[first][0], plan[first][1])] if first < len(plan) else []
            for items, questions in pending + plan[first + 1 :]:
                if self._stop.is_set():
                    raise StoppedError
                new = [i for i in items if str(i["id"]) not in held]
                if new:
                    self._set(unit.id, f"adding {len(new):,} items")
                ids = account.add(new) if new else []

                waiting = ids + [str(i["id"]) for i in items if str(i["id"]) in held]
                self._settled(account, unit.id, waiting)
                for question in questions:
                    if self._stop.is_set():
                        raise StoppedError
                    if question.id not in self.done:
                        self._set(unit.id, f"asking {asked + 1}/{len(mine)}")
                        self._write(*self._ask(account, unit, question))
                        asked += 1
        finally:
            with self._lock:
                self.added += account.added
                self.searches += account.searches
            self._set(unit.id, None)
        # Only a finished unit is erased: a stopped one keeps its memory for --resume.
        if not self.s.keep:
            account.delete()
        with self._lock:
            self._finished += 1
        progress(
            f"{unit.id}: {len(unit.items)} items, {asked} questions, "
            f"{time.monotonic() - started:.0f}s"
        )

    def _settled(self, account: Account, unit: str, ids: list[str]) -> None:
        """Wait until memory has processed `ids`. Where the API can be told the items are
        complete (`settle`), it is, once each is stored: otherwise the last of a conversation
        is processed only after it has been quiet for a while."""
        if not ids:
            return

        def storing(done: int, of: int, _seconds: float) -> None:
            self._set(unit, f"storing {done:,}/{of:,} items")

        def processing(done: int, of: int, _seconds: float) -> None:
            self._set(unit, f"processing {done:,}/{of:,} items")

        if account.settles is not False:
            account.wait(ids, self._stop, storing, until=STORED)
            account.settle()
        account.wait(ids, self._stop, processing)

    def _ask(self, account: Account, unit: Unit, question: Question) -> tuple[Json, Json]:
        row: Json = {
            "id": question.id,
            "unit": unit.id,
            "user": account.name,
            "category": question.category,
            "question": question.question,
            "answer": question.answer,
        }
        query = str(question.extra.get("query") or question.question)
        trace: Json = {"id": question.id, "unit": unit.id, "user": account.name, "query": query}
        with self._recording() as calls:
            try:
                t0 = time.monotonic()
                context, raw = account.search(query, self.s.bench.context_chars)
                row["search_seconds"] = round(time.monotonic() - t0, 2)
                row["context_chars"] = len(context)
                trace |= {"search": raw, "context": context}
                row |= self.s.bench.ask(self.models, question, context)
            except Exception as error:
                row["error"] = short(error)
                trace["error"] = traceback.format_exc(limit=5)
        trace |= {"calls": calls, "row": row}
        return row, trace

    @contextlib.contextmanager
    def _recording(self) -> Iterator[list[Json]]:
        recording = getattr(self.models, "recording", None)
        if recording is None:
            yield []
            return
        with recording() as calls:
            yield calls


def summarize(bench: Bench, variant: str, rows: list[Json]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    scores: dict[str, str] = {}
    # A benchmark that leaves failures out has nothing to score while every row failed.
    with contextlib.suppress(ZeroDivisionError):
        scores = bench.summary(rows, variant) if rows else {}
    return {
        "questions": len(rows),
        "answered": len(ok),
        "errors": len(rows) - len(ok),
        "scores": scores,
    }
