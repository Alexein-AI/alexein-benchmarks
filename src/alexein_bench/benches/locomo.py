"""LoCoMo (Maharana et al., ACL 2024; github.com/snap-research/locomo): ten long conversations
between two people over weeks, and 1,986 questions about them.

How a run goes:

- A conversation is one app user. Each session is a session of its own, each turn a message
  by its speaker, 45 seconds apart from the session's own date and time; `speaker_a` is the
  person whose memory it is. A turn that shared an image carries its caption, as LoCoMo's own
  protocol shows it: `text [shares a photo: caption]`.
- Questions are asked once every session is in memory and processed. Categories 1-4 are
  scored (1,540 questions); category 5, adversarial, is left out, as every published LoCoMo
  number leaves it out.
- The reader is gpt-4.1-mini with EverMemOS's answer prompt, verbatim, at temperature 0; the
  final answer is read off the end of its chain of thought as theirs reads it. The judge is
  gpt-4o-mini with EverMemOS's prompt, asked three times; a strict majority of CORRECT is
  correct. An empty search result is WRONG without asking the reader.
- The score is the share judged correct, over the questions that did not fail; a question
  that failed (a model error) is counted apart, as EverMemOS counts it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alexein_bench.benches import locomo_prompts as prompts
from alexein_bench.core import Json, Question, Unit
from alexein_bench.fetch import download
from alexein_bench.models import Models

REPO = "https://github.com/snap-research/locomo"
COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
URL = f"https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}/data/locomo10.json"
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

ANSWER_MODEL = "openai/gpt-4.1-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"
ANSWER_MAX_TOKENS = 32768
JUDGE_RUNS = 3
_RETRIES = 5

SCORED = (1, 2, 3, 4)
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
# Session dates read like "1:56 pm on 8 May, 2023".
_WHEN = re.compile(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})", re.I)
_MONTHS = [
    m[:3]
    for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
]
_EPOCH = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
TURN_SPACING = dt.timedelta(seconds=45)


def parse_when(value: str) -> dt.datetime | None:
    found = _WHEN.search(value or "")
    if not found:
        return None
    hour, minute, half, day, month, year = found.groups()
    if (name := month.lower()[:3]) not in _MONTHS:
        return None
    hour24 = int(hour) % 12 + (12 if half.lower() == "pm" else 0)
    return dt.datetime(
        int(year), _MONTHS.index(name) + 1, int(day), hour24, int(minute), tzinfo=dt.UTC
    )


Session = tuple[int, dt.datetime, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]
    qa: tuple[dict[str, Any], ...]


def _sessions(conversation: dict[str, Any]) -> tuple[Session, ...]:
    found = []
    for key, turns in conversation.items():
        if not re.fullmatch(r"session_\d+", key):
            continue
        number = int(key.split("_")[1])
        when = parse_when(conversation.get(f"{key}_date_time", "")) or _EPOCH + dt.timedelta(
            days=number
        )
        found.append((number, when, tuple(turns)))
    return tuple(sorted(found, key=lambda s: s[0]))


def conversations(path: Path) -> list[Conversation]:
    out: list[Conversation] = []
    for row in json.loads(path.read_text()):
        conversation = row["conversation"]
        out.append(
            Conversation(
                sample_id=str(row.get("sample_id") or f"conv-{len(out)}"),
                speaker_a=conversation.get("speaker_a", "A"),
                speaker_b=conversation.get("speaker_b", "B"),
                sessions=_sessions(conversation),
                qa=tuple(row.get("qa") or []),
            )
        )
    return out


def turn_text(turn: dict[str, Any]) -> str:
    caption = str(turn.get("blip_caption") or "").strip()
    text = str(turn.get("text") or "")
    return f"{text} [shares a photo: {caption}]" if caption else text


def _at(when: dt.datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def items(conv: Conversation, sessions: Sequence[Session] | None = None) -> list[Json]:
    """Every turn as a message, one session per LoCoMo session."""
    out: list[Json] = []
    for number, started, turns in sessions if sessions is not None else conv.sessions:
        for i, turn in enumerate(turns):
            speaker = str(turn.get("speaker") or conv.speaker_a)
            out.append(
                {
                    "id": f"{conv.sample_id}/{turn.get('dia_id') or f'S{number}:{i}'}",
                    "kind": "message",
                    "session": f"{conv.sample_id}/session_{number}",
                    "author": {
                        "id": f"{conv.sample_id}/{speaker}",
                        "name": speaker,
                        "self": speaker == conv.speaker_a,
                    },
                    "text": turn_text(turn),
                    "at": _at(started + TURN_SPACING * i),
                }
            )
    return out


def questions(conv: Conversation) -> list[Question]:
    out: list[Question] = []
    for n, q in enumerate(conv.qa):
        category = int(q.get("category", 0))
        if category not in SCORED:
            continue
        out.append(
            Question(
                id=f"{conv.sample_id}/q{n}",
                question=str(q.get("question", "")),
                answer=str(q.get("answer", q.get("adversarial_answer", ""))),
                category=CATEGORIES[category],
                extra={"speaker_a": conv.speaker_a, "speaker_b": conv.speaker_b},
            )
        )
    return out


def final_answer(text: str) -> str:
    """EverMemOS's three markers, in priority, the last occurrence winning."""
    result = text.strip()
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in result:
            answer = result.rsplit(marker, 1)[1].strip()
            if marker == "FINAL ANSWER" and answer.startswith(":"):
                answer = answer[1:].strip()
            return answer
    return result


def _label_json(content: str) -> str:
    found = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", content, re.DOTALL)
    if found:
        return found.group(1).strip()
    found = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', content)
    return found.group(0) if found else content.strip()


def label(content: str) -> str:
    """CORRECT or WRONG, as EverMemOS reads its judge; anything else raises."""
    got = str(json.loads(_label_json(content)).get("label", "")).strip().upper()
    if got not in ("CORRECT", "WRONG"):
        raise ValueError(f"unknown judge label: {got!r}")
    return got


def majority(votes: Sequence[bool]) -> bool:
    return sum(votes) > len(votes) / 2


class Locomo:
    name = "locomo"
    title = "LoCoMo"
    variants: tuple[str, ...] = ("all",)
    models: tuple[str, ...] = (ANSWER_MODEL, JUDGE_MODEL)
    tokens_per_question = 9_000
    context_chars = 24_000

    def fetch(self, cache: Path) -> Path:
        return download(URL, cache / f"locomo-{COMMIT[:7]}" / "locomo10.json", SHA256)

    def units(self, data: Path, variant: str) -> Iterator[Unit]:  # noqa: ARG002 - one variant
        for conv in conversations(data):
            yield Unit(id=conv.sample_id, items=items(conv), questions=questions(conv))

    def _answer(self, models: Models, prompt: str) -> str:
        for attempt in range(_RETRIES):
            raw = models.chat(
                ANSWER_MODEL,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=ANSWER_MAX_TOKENS,
                seed=attempt or None,
            )
            if final := final_answer(raw):
                return final
        raise RuntimeError("answer empty after retries")

    def _vote(self, models: Models, prompt: str, run: int) -> bool:
        last: Exception | None = None
        for attempt in range(_RETRIES):
            content = models.chat(
                JUDGE_MODEL,
                [
                    {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                # Three asks, not one reply read three times; a retry is a fresh ask too.
                seed=run * 100 + attempt,
            )
            try:
                return label(content) == "CORRECT"
            except (ValueError, json.JSONDecodeError, AttributeError) as error:
                last = error
        raise RuntimeError(f"judge failed after {_RETRIES} tries") from last

    def ask(self, models: Models, question: Question, context: str) -> Json:
        if not context.strip():
            return {"predicted": "[NO_CONTEXT]", "verdict": "WRONG", "score": 0.0}
        body = prompts.CONTEXT_TEMPLATE.format(
            speaker_a=question.extra["speaker_a"],
            speaker_b=question.extra["speaker_b"],
            episodes=context,
        )
        predicted = self._answer(
            models, prompts.ANSWER_PROMPT.format(context=body, question=question.question)
        )
        judge = prompts.JUDGE_USER_PROMPT.format(
            question=question.question, golden_answer=question.answer, generated_answer=predicted
        )
        votes = [self._vote(models, judge, run) for run in range(JUDGE_RUNS)]
        correct = majority(votes)
        return {
            "predicted": predicted,
            "votes": votes,
            "verdict": "CORRECT" if correct else "WRONG",
            "score": 1.0 if correct else 0.0,
        }

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        return scored(rows, variant, [CATEGORIES[c] for c in SCORED])


def scored(rows: list[Json], key: str, categories: Sequence[str]) -> dict[str, str]:
    """The mean score over rows that did not fail, overall and per category."""
    live = [r for r in rows if "error" not in r]
    if not live:
        return {}
    out = {key: f"{100 * sum(float(r['score']) for r in live) / len(live):.1f}"}
    for category in categories:
        group = [r for r in live if r.get("category") == category]
        if group:
            share = sum(float(r["score"]) for r in group) / len(group)
            out[f"{key}:{category}"] = f"{100 * share:.1f}"
    return out
