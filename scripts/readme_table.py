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


def table() -> str:
    lines = [
        "| Benchmark | What is scored | Alexein | Questions | Answers / grades | Run it |",
        "|---|---|---:|---:|---|---|",
    ]
    for board in boards():
        for row in board.rows:
            command = f"`alexein-bench run {board.bench} --variant {row.variant}`"
            grader = "no model (exact match)" if row.judge == "none" else row.judge
            lines.append(
                f"| [{board.title}]({board.source}) | {row.label} | **{row.shown}** "
                f"| {row.questions} | {row.reader} / {grader} | {command} |"
            )
    return "\n".join(lines)


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text()
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    want = f"{head}{START}\n{table()}\n{END}{tail}"
    if "--check" in sys.argv:
        if want != text:
            print("README.md's results table is out of date: python scripts/readme_table.py")
            return 1
        return 0
    readme.write_text(want)
    return 0


if __name__ == "__main__":
    sys.exit(main())
