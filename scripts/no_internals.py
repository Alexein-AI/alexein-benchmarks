"""Nothing this repository publishes may say how Alexein's memory is built inside or what it
runs on. Every tracked text file is read whole, comments included, since all of it is public.

    python scripts/no_internals.py          exit 0 clean, 3 with what was found

A benchmark's own setup is the benchmark's: the models that answer and grade, a judge, and
the published systems it is compared with (BM25, a model reading the whole history) may be
named. A benchmark's prompt is copied verbatim from its authors, so its words are theirs.
Kept in step with alexein.ai's own list (twin-website `scripts/no-internals.mjs`).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The benchmarks' own prompts, copied verbatim.
PROMPTS = re.compile(r"^src/alexein_bench/benches/\w+_prompts\.py$")
SELF = "scripts/no_internals.py"

# (pattern, may a benchmark's prompt say it)
TERMS: list[tuple[str, bool]] = [
    # The parts of memory and how it is laid out.
    (r"\bpassages?\b", True),
    (r"\bassertions?\b", True),
    (r"\blineage\b", False),
    (r"\bshards?\b", False),
    (r"\bconsumers?\b", True),
    (r"\bmanifest\b", True),
    (r"\bpolarity\b", True),
    (r"\bbi-?temporal\b", False),
    (r"\bfolded\b", True),
    (r"\blanded\b", True),
    (r"\bencode[sd]?\b(?!\()", False),
    (r"\bjev\b", False),
    (r"\bbrief\b", True),
    (r"\bstore\.|\bStore\b", False),
    (r"CORE\.md|\bcore page\b", False),
    (r"\binto records\b", False),
    (r"\bseparate database\b", False),
    (r"\bsecond guard\b", False),
    (r"[\"'`/](?:ledger|inbox|queue|thread|threads_emptied|records|window)[\"'`]", False),
    (r"\b(?:passages|records|raw|threads|inbox|assertions|ledger)/", False),
    # The engine's own ids and addresses.
    (r"\bth_[0-9a-f]{6,}", False),
    (r"\bev_[0-9a-f]{6,}", False),
    (r"\bp_\d{4}\b", False),
    (r"#r[0-9a-f]{8,}", False),
    # How an answer is found and ranked.
    (r"\brerank\w*", False),
    (r"\bembeddings?\b", True),
    (r"\bvectors?\b", True),
    (r"\bdense\b", True),
    (r"\bfusion\b|\brrf\b", False),
    (r"\brelevance check\b", False),
    (r"\bextraction\b", True),
    (r"\bknowledge graph\b", True),
    # What memory runs on. A model that answers or grades is the benchmark's, and OpenRouter is
    # where the person running it may send those calls; neither is listed.
    (r"\bdeepseek\b", False),
    (r"\bnemotron\b", False),
    (r"\bcohere\b", False),
    (r"\bmlx\b", False),
    (r"\bfp8\b|\bfp4\b|\bdeepinfra\b", False),
    (r"\b(?:sqlite|redis|litestream|postgres|fastapi|pydantic|minio)\b", False),
    (r"\btwin[-_]memory\b|\btwin_\w+", False),
    # Routes and names the public API no longer has.
    (r"/v1/(?:memory|retrieval)\b", False),
    (r"/memory/(?:overview|read|ls|lineage|recall|lessons|identify|next|unfiled)\b", False),
    (r"\bmemory_(?:overview|read|ls|recall|lessons|next)\b", False),
    (r"\bunfiled\b", False),
    (r"\bTWIN_[A-Z_]+\b", False),
    (r"\b30 minutes\b", False),
]
COMPILED = [(re.compile(t, 0 if t.startswith(r"\bTWIN") else re.IGNORECASE), ok) for t, ok in TERMS]
TEXT = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".ini", ""}


def tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return [f for f in out.stdout.splitlines() if f != SELF]


def found_in(path: str, text: str) -> list[str]:
    prompt = bool(PROMPTS.match(path))
    problems = []
    for n, line in enumerate(text.splitlines(), start=1):
        for pattern, prompt_ok in COMPILED:
            if prompt and prompt_ok:
                continue
            if match := pattern.search(line):
                problems.append(f'{path}:{n}: "{match.group(0)}" ({pattern.pattern})')
    return problems


def main() -> int:
    problems: list[str] = []
    for path in tracked():
        file = ROOT / path
        if file.suffix not in TEXT or not file.is_file():
            continue
        problems += found_in(path, file.read_text(errors="replace"))
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} found: public text says what, never how.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
