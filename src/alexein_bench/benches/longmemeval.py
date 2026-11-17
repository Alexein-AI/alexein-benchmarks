"""LongMemEval-S (Wu et al., ICLR 2025), cleaned: 500 questions, each over its own haystack of
about fifty chat sessions between a user and an assistant.

How a run goes:

- Each question is one app user, holding its haystack. Each session is a session of its own,
  named by position (`session_0`, ...); its turns are messages from `user_<question id>` and
  `assistant_<question id>`, placed 30 seconds apart from the session's own date, as EverMemOS
  places them.
- Every question is asked over its whole haystack, as LongMemEval asks it: the dataset dates 44
  questions' answer sessions on or after the question's own date, so a cutoff at that date would
  hide the answer. The question's date is given to the reader as its "Current Date".
- The reader and judge are the ones the run behind Alexein's own LongMemEval rows used:
  gemini-3.6-flash answers through EverMemOS's answer prompt, its final answer is cut from its
  chain of thought, and gpt-4o-mini judges with EverMemOS's prompt and one grading rule for the
  question's type. Neither is given a temperature: each samples at its provider's default, as
  that run's calls did.
- A search that returns nothing is never sent to the reader, and scores WRONG: an empty context
  makes every question look unanswerable, which is what the `_abs` questions test.
- The score is the share judged CORRECT. A verdict the judge never gives in four asks leaves the
  denominator, as EverMemOS's harness has it; a question that fails for any other reason scores
  WRONG.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from alexein_bench.benches import longmemeval_prompts as prompts
from alexein_bench.core import Json, Question, Unit
from alexein_bench.fetch import download
from alexein_bench.models import Models

REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    f"{REVISION}/longmemeval_s_cleaned.json"
)
SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
READER = "google/gemini-3.6-flash"
JUDGE = "openai/gpt-4o-mini"
READER_MAX_TOKENS = 32768
JUDGE_MAX_TOKENS = 512
JUDGE_ASKS = 4
EMPTY = "[NO_CONTEXT]"
TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)
# EverMemOS's date format. A date that does not parse falls back to a fixed day plus the
# session's position, which keeps the order right.
_FORMAT = "%Y/%m/%d (%a) %H:%M"
_EPOCH = dt.datetime(2023, 1, 1)
_TURN_GAP = dt.timedelta(seconds=30)


def _when(value: str, position: int) -> dt.datetime:
    try:
        return dt.datetime.strptime(value, _FORMAT)
    except (ValueError, TypeError):
        return _EPOCH + dt.timedelta(days=position)


def load(raw: dict[str, Any]) -> Unit:
    """One question and its haystack as a unit."""
    qid = str(raw["question_id"])
    items: list[Json] = []
    for position, (session, date) in enumerate(
        zip(raw["haystack_sessions"], raw["haystack_dates"], strict=True)
    ):
        start = _when(date, position)
        for ordinal, turn in enumerate(session):
            role = "user" if turn.get("role") == "user" else "assistant"
            text = str(turn.get("content") or "")
            if not text.strip():
                continue
            items.append(
                {
                    "id": f"{qid}:{position}:{ordinal}",
                    "kind": "message",
                    "session": f"session_{position}",
                    "role": role,
                    "author": f"{role}_{qid}",
                    "text": text,
                    "at": (start + _TURN_GAP * ordinal).isoformat(timespec="seconds"),
                }
            )
    question = Question(
        id=qid,
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        category=str(raw["question_type"]),
        cutoff=None,
        date=str(raw.get("question_date") or ""),
        extra={"abstention": qid.endswith("_abs")},
    )
    return Unit(id=qid, items=items, questions=[question])


class LongMemEval:
    name = "longmemeval"
    title = "LongMemEval-S"
    variants: tuple[str, ...] = ("all",)
    models: tuple[str, ...] = (READER, JUDGE)
    tokens_per_question = 12_000
    context_chars = 24_000

    def fetch(self, cache: Path) -> Path:
        return download(URL, cache / f"longmemeval_s_cleaned-{REVISION[:7]}.json", SHA256)

    def units(self, data: Path, variant: str) -> Iterator[Unit]:  # noqa: ARG002 - one variant
        for raw in json.loads(data.read_text()):
            yield load(raw)

    def ask(self, models: Models, question: Question, context: str) -> Json:
        if not context.strip():
            return {"predicted": EMPTY, "verdict": "WRONG", "score": 0.0}
        speakers = {"speaker_a": f"user_{question.id}", "speaker_b": f"assistant_{question.id}"}
        shown = prompts.CONTEXT_TEMPLATE.format(**speakers, episodes=context)
        date_line = f"Current Date: {question.date}\n" if question.date else ""
        prompt = prompts.ANSWER_PROMPT.format(
            context=shown, current_date_line=date_line, question=question.question
        )
        raw = models.chat(
            READER,
            [{"role": "user", "content": prompt}],
            temperature=None,
            max_tokens=READER_MAX_TOKENS,
        ).strip()
        predicted = prompts.extract_final_answer(raw)
        verdict = self._judge(models, question, predicted)
        row: Json = {"predicted": predicted, "verdict": verdict or "UNJUDGED"}
        if verdict is not None:
            row["score"] = 1.0 if verdict == "CORRECT" else 0.0
        return row

    def _judge(self, models: Models, question: Question, predicted: str) -> str | None:
        clause = prompts.leniency_clause(question.id, question.category)
        body = prompts.JUDGE_USER_PROMPT.format(
            question=question.question, golden_answer=question.answer, generated_answer=predicted
        )
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": clause + body},
        ]
        for attempt in range(JUDGE_ASKS):
            reply = models.chat(
                JUDGE, messages, temperature=None, max_tokens=JUDGE_MAX_TOKENS, seed=attempt
            )
            if (label := prompts.parse_judge_label(reply)) is not None:
                return label
        return None

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        # A row that failed scores WRONG; one the judge never read leaves the denominator.
        judged = [r for r in rows if "error" in r or "score" in r]
        if not judged:
            return {}
        out = {variant: _share(judged)}
        for kind in TYPES:
            group = [r for r in judged if r.get("category") == kind]
            if group:
                out[f"{variant}:{kind}"] = _share(group)
        out[f"{variant}:unjudged"] = str(len(rows) - len(judged))
        return out


def _share(rows: list[Json]) -> str:
    return f"{100 * sum(float(r.get('score', 0.0)) for r in rows) / len(rows):.1f}"
