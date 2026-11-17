"""MemoryAgentBench (Hu et al., arXiv 2507.05257): one long context fed to a memory a chunk at
a time, then many questions asked of it ("inject once, query many times").

Three of its datasets, the three variants here:

- `fc`: fact consolidation, single-hop, 262K tokens - 1 context, 100 questions. A later fact
  overwrites an earlier one; the newest wins. Each numbered fact is one item, its serial kept in
  its text: the benchmark's own prompt defines "newer" as "larger serial number".
- `ruler`: RULER-QA1, 197K tokens - 1 context, 100 questions. Accurate retrieval.
- `detective`: DetectiveQA, all 10 books - 71 questions. Long-range understanding.

How a run goes, as the published numbers were produced:

- A context is one app user. RULER-QA1 is cut into windows of about 512 tokens, DetectiveQA
  into windows of about 4,096 (the paper's chunk sizes), sentences packed whole at four
  characters a token. Every chunk is a message from the user, a minute after the one before it
  from 2025-01-01, so memory's order is the benchmark's order. Every question is asked once the
  whole context is processed.
- The search is the question as MemoryAgentBench's own retrieval reads it: the part after "Now
  Answer the Question:" when there is one.
- The reader is gpt-4.1-mini with MemoryAgentBench's system message, its `rag_agent` query
  template after the search result, its answer length per dataset, at temperature 0.7.
- The score is MemoryAgentBench's (`utils/eval_other_utils.py`): substring exact match for fact
  consolidation and RULER-QA, exact match for DetectiveQA, each the better of the raw and the
  parsed answer; DetectiveQA's answer is read from the JSON its prompt asks for. A question that
  failed to run is left out of the mean, as the published run left it out; the summary counts
  them.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import string
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexein_bench.benches import mab_prompts as prompts
from alexein_bench.core import Json, Question, Unit
from alexein_bench.fetch import download
from alexein_bench.models import Models

REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
_RESOLVE = f"https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/{REVISION}/data"
MODEL = "openai/gpt-4.1-mini"


@dataclass(frozen=True)
class Subset:
    source: str
    split: str
    family: str
    # The headline metric, as the benchmark's README maps it.
    metric: str
    # Sentence-packed window size in tokens; None cuts one chunk per numbered fact.
    window: int | None
    sha256: str


SUBSETS = {
    "fc": Subset(
        "factconsolidation_sh_262k",
        "Conflict_Resolution",
        "factconsolidation",
        "substring_exact_match",
        None,
        "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
    ),
    "ruler": Subset(
        "ruler_qa1_197K",
        "Accurate_Retrieval",
        "ruler_qa",
        "substring_exact_match",
        512,
        "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
    ),
    "detective": Subset(
        "detective_qa",
        "Long_Range_Understanding",
        "detective_qa",
        "exact_match",
        4096,
        "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
    ),
}
# The usual English estimate for the benchmark's tokenizer, for window budgets only.
CHARS_PER_TOKEN = 4
_FACT = re.compile(r"^(\d+)\. (.+)$")
_SENTENCE = re.compile(r"(?<=[.!?”\"])\s+")
# Its `_extract_retrieval_query`: what its retrieval searches for is the part after this.
_ASK = "Now Answer the Question:"
_BASE = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
_SPACING = dt.timedelta(minutes=1)


# --- Chunks -----------------------------------------------------------------------------------


def facts(text: str) -> list[str]:
    """'N. fact' lines, each kept whole, serial included."""
    return [line.strip() for line in text.splitlines() if _FACT.match(line.strip())]


def windows(text: str, tokens: int) -> list[str]:
    """Sentences packed into windows of about `tokens`, the way the benchmark packs them."""
    budget = tokens * CHARS_PER_TOKEN
    out: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in _SENTENCE.split(text.strip()):
        if current and size + len(sentence) > budget:
            out.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        out.append(" ".join(current))
    return out


def query_of(question: str) -> str:
    _, marker, rest = question.partition(_ASK)
    return rest.strip() if marker else question.strip()


def _at(values: list[Any] | None, n: int) -> str:
    return str(values[n]) if values and n < len(values) else ""


def unit_of(row: Json, position: int, subset: Subset) -> Unit:
    text = str(row["context"])
    chunks = facts(text) if subset.window is None else windows(text, subset.window)
    uid = f"{subset.source}-{position}"
    items = [
        {
            "id": f"{uid}:{n}",
            "kind": "message",
            "role": "user",
            "text": chunk,
            "at": (_BASE + _SPACING * n).isoformat().replace("+00:00", "Z"),
        }
        for n, chunk in enumerate(chunks)
    ]
    meta = row.get("metadata") or {}
    questions = [
        Question(
            id=f"{uid}/{_at(meta.get('qa_pair_ids'), n) or n}",
            question=str(q),
            answer=json.dumps([str(a) for a in answers], ensure_ascii=False),
            category=_at(meta.get("question_types"), n) or subset.family,
            extra={
                "answers": [str(a) for a in answers],
                "family": subset.family,
                "metric": subset.metric,
                # What memory is searched with; the reader still sees the whole question.
                "query": query_of(str(q)),
            },
        )
        for n, (q, answers) in enumerate(zip(row["questions"], row["answers"], strict=True))
    ]
    return Unit(id=uid, items=items, questions=questions)


# --- Scoring, ported line for line from `utils/eval_other_utils.py` ---------------------------

_SPECIAL = frozenset({"yes", "no", "noanswer"})


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(prediction: str, truth: str) -> float:
    guess, gold = normalize_answer(prediction), normalize_answer(truth)
    if (guess in _SPECIAL or gold in _SPECIAL) and guess != gold:
        return 0.0
    common = sum((Counter(guess.split()) & Counter(gold.split())).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(guess.split()), common / len(gold.split())
    return 2 * precision * recall / (precision + recall)


def calculate(prediction: str, answers: list[str]) -> dict[str, float]:
    """`calculate_metrics` without ROUGE: each metric's best over the gold answers."""
    norm = normalize_answer(prediction)
    return {
        "exact_match": max(float(norm == normalize_answer(a)) for a in answers),
        "f1": max(f1(prediction, a) for a in answers),
        "substring_exact_match": max(float(normalize_answer(a) in norm) for a in answers),
    }


def parse_output(output: str, prefix: str = "Answer:") -> str | None:
    for pattern in (
        re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ):
        if found := pattern.search(output):
            return re.sub(
                f"^{re.escape(prefix)}", "", found[1].strip(), flags=re.IGNORECASE
            ).strip()
    return None


_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"([^"]*)"')


def post_process(output: str, answers: list[str], family: str) -> dict[str, float]:
    """`post_process`: the better of the raw output and the parsed one, metric by metric.
    DetectiveQA's answer is read from the JSON its prompt asks for."""
    if family == "detective_qa" and (field := _ANSWER_FIELD.search(output)):
        output = field[1]
    metrics = calculate(output, answers)
    parsed = parse_output(output)
    if parsed is not None:
        again = calculate(parsed, answers)
        metrics = {name: max(value, again[name]) for name, value in metrics.items()}
    return metrics


# --- The benchmark ----------------------------------------------------------------------------


def _rows(path: Path, source: str) -> Iterator[Json]:
    with path.open() as handle:
        for line in handle:
            row: Json = json.loads(line)
            if (row.get("metadata") or {}).get("source") == source:
                yield row


def _to_jsonl(parquet: Path) -> Path:
    target = parquet.with_suffix(".jsonl")
    if target.exists():
        return target
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - only fetching needs it
    except ImportError as error:
        raise SystemExit(
            "MemoryAgentBench ships as parquet, which needs pyarrow: `pip install pyarrow`."
        ) from error
    part = target.with_suffix(".part")
    with part.open("w") as handle:
        for row in pq.read_table(parquet).to_pylist():
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    part.replace(target)
    return target


class Mab:
    name = "mab"
    title = "MemoryAgentBench"
    variants: tuple[str, ...] = ("fc", "ruler", "detective")
    models: tuple[str, ...] = (MODEL,)
    tokens_per_question = 4_000
    context_chars = 24_000

    def fetch(self, cache: Path) -> Path:
        root = cache / f"mab-{REVISION[:7]}"
        for subset in SUBSETS.values():
            parquet = root / f"{subset.split}.parquet"
            url = f"{_RESOLVE}/{subset.split}-00000-of-00001.parquet"
            _to_jsonl(download(url, parquet, subset.sha256))
        return root

    def units(self, data: Path, variant: str) -> Iterator[Unit]:
        subset = SUBSETS[variant]
        for position, row in enumerate(_rows(data / f"{subset.split}.jsonl", subset.source)):
            yield unit_of(row, position, subset)

    def ask(self, models: Models, question: Question, context: str) -> Json:
        family = str(question.extra["family"])
        prompt = context + "\n" + prompts.RAG_QUERY[family].format(question=question.question)
        predicted = models.chat(
            MODEL,
            [{"role": "system", "content": prompts.SYSTEM}, {"role": "user", "content": prompt}],
            temperature=prompts.TEMPERATURE,
            max_tokens=prompts.MAX_TOKENS[family],
        )
        metrics = post_process(predicted, list(question.extra["answers"]), family)
        score = metrics[str(question.extra["metric"])]
        return {"predicted": predicted, "score": score, "metrics": metrics}

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        scored = [r for r in rows if "error" not in r]
        if not scored:
            return {}
        mean = sum(float(r["score"]) for r in scored) / len(scored)
        return {variant: f"{100 * mean:.1f}"}
