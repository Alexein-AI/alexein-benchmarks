"""What every benchmark hands the runner: units of memory, each with the items it holds and the
questions asked of it, and how to answer and score one question.

A unit is one app user's memory: a LoCoMo conversation, a LongMemEval question's haystack, a
RHELM character. Its items go into Alexein in time order, and each question is asked once
everything dated before its `cutoff` is in memory and processed - never anything after, so a
question sees only what was said before the day it is asked.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from alexein_bench.models import Models

Json = dict[str, Any]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    category: str
    # ISO date or datetime. Items whose `at` is earlier are in memory when this is asked; None
    # asks after every item. `date` is the question's own date, as its prompt states it.
    cutoff: str | None = None
    date: str | None = None
    # Anything else the benchmark keeps. `query`, when set, is what is searched in place of
    # the question's whole text.
    extra: Json = field(default_factory=dict)


@dataclass(frozen=True)
class Unit:
    id: str
    # Each an item for `POST /v2/memory/add`: text, kind, id, at, role, author, session, title,
    # metadata. `at` is an ISO datetime and every item has one.
    items: list[Json]
    questions: list[Question]


class Bench(Protocol):
    name: str
    title: str
    # Variants of one benchmark that are scored apart: RHELM's two leaderboard tables.
    variants: tuple[str, ...]
    # Models it needs, as OpenRouter names them (`openai/gpt-4.1-mini`).
    models: tuple[str, ...]
    # Rough tokens per question across every model call, for the estimate shown before a run.
    tokens_per_question: int
    # Characters of search context given to the reader.
    context_chars: int

    def fetch(self, cache: Path) -> Path:
        """Download the dataset (once) and return where it is."""
        ...

    def units(self, data: Path, variant: str) -> Iterator[Unit]: ...

    def ask(self, models: Models, question: Question, context: str) -> Json:
        """Answer from `context` and score it: a row with at least `predicted` and `score`
        (0 to 1). Anything else the benchmark keeps goes in the row too."""
        ...

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        """The headline numbers over every row asked - a row with `error` counts as the run
        behind the published number counted one: RHELM's as 0, MemoryAgentBench's left out -
        each written as the website writes it and keyed by its
        row's `key` in `results/<bench>.json`."""
        ...
