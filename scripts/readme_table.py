"""The README's results table, written from `results/*.json` so the two never disagree.

python scripts/readme_table.py          rewrite the table in README.md
python scripts/readme_table.py --check  exit 1 if README.md is out of date
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alexein_bench.published import boards  # noqa: E402

START, END = "<!-- results:start -->", "<!-- results:end -->"


def heading(title: str) -> str:
    """The README heading a benchmark's chart and full commands sit under: "BEAM, 1M-token
    split" is under "### BEAM"."""
    return title.split(",", maxsplit=1)[0]


def anchor(title: str) -> str:
    """GitHub's link to that heading: "LongMemEval-S" is #longmemeval-s."""
    return heading(title).lower().replace(" ", "-")


def table() -> str:
    # A benchmark and what it scores share a cell, and the commands, too wide for GitHub's
    # column, are a link to the benchmark's own heading, where they are written out in full.
    lines = [
        "| Benchmark | Alexein | Questions | Answers / grades | Run it |",
        "|---|---:|---:|---|---|",
    ]
    for board in boards():
        for row in board.rows:
            command = f"[Command](#{anchor(board.title)})"
            grader = "no model (exact match)" if row.judge == "none" else row.judge
            lines.append(
                f"| [{board.title}]({board.source})<br>{row.label} | **{row.shown}** "
                f"| {row.questions} | {row.reader} / {grader} | {command} |"
            )
    return "\n".join(lines)


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text()
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    want = f"{head}{START}\n{table()}\n{END}{tail}"
    # Each row's command links to its benchmark's heading, so the heading must be there.
    missing = [b.title for b in boards() if f"\n### {heading(b.title)}\n" not in text]
    for title in missing:
        print(f"README.md has no '### {heading(title)}' for {title}'s commands to link to")
    if missing:
        return 1
    if "--check" in sys.argv:
        if want != text:
            print("README.md's results table is out of date: python scripts/readme_table.py")
            return 1
        return 0
    readme.write_text(want)
    return 0


if __name__ == "__main__":
    sys.exit(main())
