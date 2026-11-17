"""The numbers alexein.ai/benchmarks shows, read from `results/<bench>.json`: each written as the
website writes it, with the setup that produced it and whether this harness repeats that setup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    key: str
    variant: str
    label: str
    shown: str
    metric: str
    questions: str
    reader: str
    judge: str
    # "reproducible": the run behind it is known and this harness repeats its setup.
    # "pending": shown on the website; the run behind it is not yet published here.
    status: str
    note: str = ""


@dataclass(frozen=True)
class Board:
    bench: str
    title: str
    source: str
    rows: list[Row]


def _root() -> Path:
    here = Path(__file__).resolve().parents[2] / "results"
    if here.is_dir():
        return here
    return Path(str(resources.files("alexein_bench") / "results"))


def board(bench: str) -> Board:
    raw: dict[str, Any] = json.loads((_root() / f"{bench}.json").read_text())
    return Board(
        bench=raw["bench"],
        title=raw["title"],
        source=raw["source"],
        rows=[Row(**r) for r in raw["rows"]],
    )


# As the README lists them: those this harness reproduces first.
ORDER = ("rhelm", "mab", "locomo", "locomo_plus", "longmemeval", "beam")


def boards() -> list[Board]:
    names = [p.stem for p in _root().glob("*.json")]
    names.sort(key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    return [board(n) for n in names]
