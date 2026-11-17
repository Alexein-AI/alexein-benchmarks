"""BEAM (Tavakoli et al., arXiv 2510.27246), its 1M-token split: 35 conversations of about a
million tokens each, every one larger than GPT-5's context window, and 700 probing questions -
two of each of ten abilities per conversation.

Scored as Mem0's published harness scores it (github.com/mem0ai/memory-benchmarks, Apache
License 2.0, `benchmarks/beam/run.py` at commit 4b61c5d), which is how the website's number and
Mem0's own (64.09 mean, 70.14% pass) were both produced:

- The dataset is Hugging Face's `Mohammadta/BEAM`, split `1M`, the copy their harness reads,
  pinned to the revision of that time and checked by digest.
- A conversation is one app user. Its chat is batches of messages; each batch carries a date
  (`time_anchor`, "March-01-2024"). Every message is an item of its own, its batch's date plus
  a minute per message, so the order holds; each batch is a session. Empty messages are skipped,
  as theirs skips them. Every question is asked after the whole conversation is in memory.
- GPT-5 answers and grades, called as their client calls it: no temperature (GPT-5 takes only
  its default), at most 4,096 completion tokens, no system message for the answer, the judge's
  system message and JSON mode for the grade. A reply that comes back empty, or a grade that
  does not read as JSON, is asked again, up to five times, as theirs retries. GPT-5 samples,
  so no reply is read back from a cache: a run is one draw, as theirs is.
- Each rubric item is graded alone, 0, 0.5 or 1 (a raw score rounds to 1 from 0.75 and to 0.5
  from 0.25). A question's score is the mean over its rubric, rounded to four places.
- `mean` is the mean question score, as a percentage; `pass` the share of questions scoring at
  least 0.5. A question that fails scores 0 and still counts, as theirs counts it.

Not repeated: their extra Kendall tau-b pass on event-ordering questions, which neither
headline number reads.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from alexein_bench.benches import beam_prompts as prompts
from alexein_bench.core import Json, Question, Unit
from alexein_bench.fetch import download
from alexein_bench.models import Models

DATASET = "Mohammadta/BEAM"
REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
SPLIT = "1M"
PARQUET_SHA256 = "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{SPLIT}-00000-of-00001.parquet"
MODEL = "openai/gpt-5"
MAX_TOKENS = 4096
ATTEMPTS = 5
PASS = 0.5
# Their question order: ten abilities, two questions each.
ABILITIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
_ANCHOR = "%B-%d-%Y"
_FIRST_DAY = dt.datetime(2024, 1, 1)


def _convert(parquet: Path, target: Path) -> None:
    """The parquet's conversations as JSON: read once, at download, with pyarrow."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - only at download
    except ImportError as error:
        raise SystemExit(
            "BEAM ships as parquet: install pyarrow once to read it (pip install pyarrow)."
        ) from error
    rows = pq.read_table(parquet).to_pylist()
    kept = [
        {
            "conversation_id": str(r["conversation_id"]),
            "chat": r["chat"],
            "probing_questions": r["probing_questions"],
        }
        for r in rows
    ]
    part = target.with_suffix(".part")
    part.write_text(json.dumps(kept, ensure_ascii=False))
    part.replace(target)


def _anchor(batch: list[Json], previous: dt.datetime | None) -> dt.datetime:
    """The batch's date: its first message's `time_anchor`, else the day after the last."""
    for message in batch:
        if message.get("time_anchor"):
            return dt.datetime.strptime(str(message["time_anchor"]), _ANCHOR)
    return previous + dt.timedelta(days=1) if previous else _FIRST_DAY


def items_of(conversation: Json) -> list[Json]:
    items: list[Json] = []
    day: dt.datetime | None = None
    for b, batch in enumerate(conversation["chat"]):
        day = _anchor(batch, day)
        for k, message in enumerate(batch):
            text = str(message.get("content") or "")
            if not text.strip():
                continue
            role = str(message.get("role") or "user")
            if role not in ("user", "assistant"):
                role = "user" if role.lower() in ("human", "user") else "assistant"
            items.append(
                {
                    "id": f"beam-1m/{conversation['conversation_id']}/{b}/{k}",
                    "kind": "message",
                    "session": f"batch-{b + 1}",
                    "role": role,
                    "author": "User" if role == "user" else "Assistant",
                    "text": text,
                    "at": (day + dt.timedelta(minutes=k)).isoformat(),
                }
            )
    return items


def _rubric(question: Json) -> list[str]:
    """`extract_rubric_nuggets`: a list as given, a dict's `nuggets`, else the one value."""
    raw = question.get("rubric", {})
    if isinstance(raw, dict):
        return [
            str(n.get("description", n)) if isinstance(n, dict) else str(n)
            for n in raw.get("nuggets", [])
        ]
    if isinstance(raw, list):
        return [str(n) for n in raw]
    return [str(raw)] if raw else []


def questions_of(conversation: Json) -> list[Question]:
    raw = conversation["probing_questions"]
    probing: Json = ast.literal_eval(raw) if isinstance(raw, str) else raw
    out: list[Question] = []
    for ability in ABILITIES:
        for n, q in enumerate(probing.get(ability, [])):
            rubric = _rubric(q)
            out.append(
                Question(
                    id=f"{conversation['conversation_id']}/{ability}/{n}",
                    question=str(q.get("question_text", q.get("question", ""))),
                    answer=" | ".join(rubric),
                    category=ability,
                    extra={"rubric": rubric, "difficulty": q.get("difficulty", "unknown")},
                )
            )
    return out


def clamp(raw: float) -> float:
    if raw >= 0.75:
        return 1.0
    if raw >= 0.25:
        return 0.5
    return 0.0


def grade(reply: Any) -> float | None:
    """A judge reply's score, read as `judge_single_nugget` reads it; None when it does not
    read as JSON, which theirs asks again."""
    try:
        parsed = json.loads(str(reply).strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict) and "final" in parsed and len(parsed) == 1:
        inner = parsed["final"]
        try:
            parsed = json.loads(inner) if isinstance(inner, str) else inner
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(parsed, dict):
        try:
            return clamp(float(parsed.get("score", 0.0)))
        except (ValueError, TypeError):
            return 0.0
    text = str(parsed)
    return 1.0 if "1.0" in text else 0.5 if "0.5" in text else 0.0


class Beam:
    name = "beam"
    title = "BEAM, 1M-token split"
    variants: tuple[str, ...] = ("1m",)
    models: tuple[str, ...] = (MODEL,)
    tokens_per_question = 30_000
    context_chars = 24_000

    def fetch(self, cache: Path) -> Path:
        folder = cache / f"beam-{REVISION[:7]}"
        target = folder / f"{SPLIT}.json"
        if not target.exists():
            parquet = download(URL, folder / f"{SPLIT}.parquet", PARQUET_SHA256)
            _convert(parquet, target)
        return target

    def units(self, data: Path, variant: str) -> Iterator[Unit]:  # noqa: ARG002 - one variant
        for conversation in json.loads(data.read_text()):
            yield Unit(
                id=f"1m-{conversation['conversation_id']}",
                items=items_of(conversation),
                questions=questions_of(conversation),
            )

    def _answer(self, models: Models, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for _ in range(ATTEMPTS):
            text = models.chat(
                MODEL, messages, temperature=None, max_tokens=MAX_TOKENS, reuse=False
            ).strip()
            if text:
                return text
        return ""

    def _grade(self, models: Models, prompt: str) -> float:
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        for _ in range(ATTEMPTS):
            reply = models.chat(
                MODEL,
                messages,
                temperature=None,
                max_tokens=MAX_TOKENS,
                json_object=True,
                reuse=False,
            )
            score = grade(reply) if reply.strip() else None
            if score is not None:
                return score
        return 0.0

    def ask(self, models: Models, question: Question, context: str) -> Json:
        prompt = prompts.ANSWER_PROMPT.format(
            question=question.question, memories_text=context or prompts.NO_MEMORIES
        )
        predicted = self._answer(models, prompt)
        if "ANSWER:" in predicted:
            predicted = predicted.rsplit("ANSWER:", 1)[-1].strip()
        rubric: list[str] = list(question.extra.get("rubric", []))
        if not rubric:
            return {"predicted": predicted, "score": 0.0, "nuggets": [], "unread": "no rubric"}
        nuggets = [
            self._grade(
                models,
                prompts.JUDGE_PROMPT.format(
                    question=question.question, llm_response=predicted, nugget=nugget
                ),
            )
            for nugget in rubric
        ]
        score = round(sum(nuggets) / len(nuggets), 4)
        return {"predicted": predicted, "score": score, "nuggets": nuggets}

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:  # noqa: ARG002
        scores = [float(r.get("score", 0.0)) for r in rows]
        out = {
            "mean": f"{100 * sum(scores) / len(scores):.2f}",
            "pass": f"{100 * sum(s >= PASS for s in scores) / len(scores):.2f}%",
        }
        for ability in ABILITIES:
            group = [float(r.get("score", 0.0)) for r in rows if r.get("category") == ability]
            if group:
                out[f"mean:{ability}"] = f"{100 * sum(group) / len(group):.2f}"
        return out
