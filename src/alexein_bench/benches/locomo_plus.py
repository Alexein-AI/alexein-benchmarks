"""LoCoMo-Plus (Li et al., arXiv 2602.10715; github.com/xjtuleeyf/Locomo-Plus): 401 short cue
dialogues - causal, state, goal and value - each stitched into a LoCoMo conversation, and a
one-line trigger spoken weeks or months later that depends on the cue without repeating it.
There is no gold answer: a judge reads the cue and asks whether the reply shows it was
remembered.

How a run goes:

- Each LoCoMo conversation is one app user: the conversation, and every cue placed in it (item
  i joins conversation i mod 10, so about 40 each), each cue a session of its own. Placement is
  the paper's (`data/build_conv.py`): the trigger comes a week after the last session, and the
  cue `time_gap` before that. Their gap parser is kept as it is, so a gap it cannot read
  ("several months later") counts as none. The paper stitches one cue into a conversation at a
  time; here a conversation's cues share one memory, so each question's history also holds the
  other cues of its conversation - ten app users and about 8,000 items, where one per question
  is 401 and 236,000. Every question is asked once all of its conversation is in memory.
- The trigger is not added. Once the conversation is processed, memory is searched with the
  trigger's words, and the reader - gpt-4.1-mini at temperature 0.3, up to 1,024 tokens, the
  settings of their evaluate.sh - gets their opening line and instruction, the search result
  where their stitched conversation goes, and the trigger last, in their `said, "..."` form.
- The judge is the paper's, Gemini 2.5 Flash at temperature 0, with their Cognitive prompt.
  Its label is read as their `_parse_judge_response` reads it, quirks kept: correct 1,
  partial 0.5, wrong 0. A question that fails is left out of the mean.
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from alexein_bench.benches import locomo
from alexein_bench.benches import locomo_plus_prompts as prompts
from alexein_bench.core import Json, Question, Unit
from alexein_bench.fetch import download
from alexein_bench.models import Models

REPO = "https://github.com/xjtuleeyf/Locomo-Plus"
COMMIT = "059f4e3d38f7f1f96765e8e2cb7de3097551bffb"
URL = f"https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/{COMMIT}/data/locomo_plus.json"
SHA256 = "af2fc6e5e0e88735e28d8dd1ef474b718861a73beb21779809e3b9e1a4fe3d23"

READER_MODEL = "openai/gpt-4.1-mini"
READER_TEMPERATURE = 0.3
READER_MAX_TOKENS = 1024
JUDGE_MODEL = "google/gemini-2.5-flash"
JUDGE_MAX_TOKENS = 512
SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
RELATIONS = ("causal", "state", "goal", "value")
ASKED_AFTER = dt.timedelta(days=7)
# The full-width colon their _cue_dialogue_to_evidence writes between a speaker and a line.
FULL_COLON = chr(0xFF1A)
# Their parse_time_gap's words, in their order.
_NUMBERS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
_COUNTS = {w: n for n, w in enumerate((*_NUMBERS, "eleven", "twelve"), 1)} | {"a": 1, "an": 1}
_GAP = re.compile(rf"\b(\d+|{'|'.join(_COUNTS)})\b\s*(week|weeks|month|months|year|years)\b")
_DAYS = {"week": 7, "month": 30, "year": 365}


def gap_days(time_gap: str) -> int:
    """Their parse_time_gap: a week is 7 days, a month 30, a year 365, anything else 0."""
    found = _GAP.search(time_gap.lower().strip())
    if not found:
        return 0
    count, unit = found.groups()
    return (int(count) if count.isdigit() else _COUNTS[count]) * _DAYS[unit.rstrip("s")]


def _lines(text: str) -> tuple[tuple[str, str], ...]:
    """Their parse_ab_dialogue: every line that opens "A:" or "B:", as (letter, text)."""
    found = (line.strip() for line in text.split("\n"))
    return tuple((line[0], line[2:].strip()) for line in found if line.startswith(("A:", "B:")))


@dataclass(frozen=True)
class Item:
    index: int
    conversation: int
    relation: str
    cue: tuple[tuple[str, str], ...]
    trigger: tuple[tuple[str, str], ...]
    time_gap: str

    def speaker(self, letter: str, conv: locomo.Conversation) -> str:
        """Their map_speaker: A is speaker_a, anything else speaker_b."""
        return conv.speaker_a if letter == "A" else conv.speaker_b

    def query(self) -> str:
        return " ".join(text for _, text in self.trigger)

    def evidence(self, conv: locomo.Conversation) -> str:
        """What their judge is shown: the cue, a line a speaker, by name, after the full-width
        colon their _cue_dialogue_to_evidence writes."""
        return "\n".join(
            f"{self.speaker(a, conv)}{FULL_COLON}{text}" for a, text in self.cue if text
        )


def load(path: Path, *, conversations: int = 10) -> list[Item]:
    return [
        Item(
            index=i,
            conversation=i % conversations,
            relation=str(row.get("relation_type", "")),
            cue=_lines(str(row.get("cue_dialogue", ""))),
            trigger=_lines(str(row.get("trigger_query", ""))),
            time_gap=str(row.get("time_gap", "")),
        )
        for i, row in enumerate(json.loads(path.read_text()))
    ]


def cue_at(item: Item, conv: locomo.Conversation) -> tuple[dt.datetime, int]:
    """Their compute_insertion: when the cue is said, and how many sessions come before it."""
    asked_at = conv.sessions[-1][1] + ASKED_AFTER
    when = asked_at - dt.timedelta(days=gap_days(item.time_gap))
    after = bisect.bisect_right([started for _, started, _ in conv.sessions], when)
    return when, after


def stitched(cues: list[Item], conv: locomo.Conversation) -> list[locomo.Session]:
    """The conversation's sessions with each cue in as one more, at its place: in time order,
    a cue after any session that starts when it does, as their insertion puts it."""
    sessions = list(conv.sessions)
    number = max(n for n, _, _ in conv.sessions)
    for item in cues:
        when, _ = cue_at(item, conv)
        number += 1
        turns = tuple(
            {"speaker": item.speaker(letter, conv), "dia_id": f"C{item.index}:{n}", "text": text}
            for n, (letter, text) in enumerate(item.cue, start=1)
        )
        sessions.append((number, when, turns))
    return sorted(sessions, key=lambda session: session[1])


def question(item: Item, conv: locomo.Conversation) -> Question:
    return Question(
        id=f"plus-{item.index}",
        question=item.query(),
        answer=item.evidence(conv),
        category=item.relation,
        extra={
            "speaker_a": conv.speaker_a,
            "speaker_b": conv.speaker_b,
            "said": "\n".join(f'{item.speaker(a, conv)} said, "{t}"' for a, t in item.trigger),
            "time_gap": item.time_gap,
        },
    )


def unit(cues: list[Item], conv: locomo.Conversation) -> Unit:
    """One conversation's app user: it, every cue placed in it, and their questions."""
    return Unit(
        id=f"plus-{conv.sample_id}",
        items=locomo.items(conv, stitched(cues, conv)),
        questions=[question(item, conv) for item in cues],
    )


_LABELLED = re.compile(
    r'\{[^{}]*"label"\s*:\s*["\']([^"\']+)["\'][^{}]*"reason"\s*:\s*["\']([^"\']*)["\']', re.DOTALL
)


def label(raw: str) -> str:
    """Their _parse_judge_response reduced to the label, quirks kept: an object with "label"
    before "reason", else the reply as JSON, else - only when that fails to parse - the first
    of correct, wrong, partial the reply contains, so "incorrect" counts as correct there."""
    text = (raw or "").strip()
    if found := _LABELLED.search(text):
        return found.group(1).strip().lower()
    try:
        return str(json.loads(text).get("label") or "").strip().lower()
    except (ValueError, AttributeError):
        lowered = text.lower()
        return next((word for word in ("correct", "wrong", "partial") if word in lowered), "")


def reader_prompt(context: str, question: Question) -> str:
    """Their _build_model_input with Alexein's search result where their stitched conversation
    goes and the trigger last. The opening line names the speakers; theirs leaves its defaults,
    "A" and "B", above a dialogue that uses names."""
    body = f"{context.strip()}\n\n{question.extra['said']}".strip()
    start = prompts.CONV_START_PROMPT.format(
        question.extra["speaker_a"], question.extra["speaker_b"]
    )
    return start + prompts.INSTRUCTION_COGNITIVE + body


def judge_prompt(evidence: str, prediction: str) -> str:
    """Exactly what their judge is sent: their call_llm puts CONV_START_PROMPT, for speakers
    "A" and "B", in front of every prompt, the judge's included."""
    filled = prompts.COGNITIVE_JUDGE.format(pred=prediction or "", evidence=evidence or "")
    return prompts.CONV_START_PROMPT.format("A", "B") + filled.strip()


class LocomoPlus:
    name = "locomo_plus"
    title = "LoCoMo-Plus"
    variants: tuple[str, ...] = ("all",)
    models: tuple[str, ...] = (READER_MODEL, JUDGE_MODEL)
    tokens_per_question = 8_000
    context_chars = 24_000

    def fetch(self, cache: Path) -> Path:
        folder = cache / f"locomo-plus-{COMMIT[:7]}"
        download(locomo.URL, folder / "locomo10.json", locomo.SHA256)
        download(URL, folder / "locomo_plus.json", SHA256)
        return folder

    def units(self, data: Path, variant: str) -> Iterator[Unit]:  # noqa: ARG002 - one variant
        convs = locomo.conversations(data / "locomo10.json")
        cues = load(data / "locomo_plus.json", conversations=len(convs))
        for n, conv in enumerate(convs):
            yield unit([item for item in cues if item.conversation == n], conv)

    def ask(self, models: Models, question: Question, context: str) -> Json:
        reply = models.chat(
            READER_MODEL,
            [{"role": "user", "content": reader_prompt(context, question)}],
            temperature=READER_TEMPERATURE,
            max_tokens=READER_MAX_TOKENS,
        )
        # Their call_llm hands an empty completion on as "(empty)", and so the judge sees it.
        predicted = reply.strip() or "(empty)"
        judged = models.chat(
            JUDGE_MODEL,
            [{"role": "user", "content": judge_prompt(question.answer, predicted)}],
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        if not judged.strip():
            raise ValueError("the judge returned nothing")
        got = label(judged)
        return {"predicted": predicted, "label": got, "score": SCORES.get(got, 0.0)}

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        return locomo.scored(rows, variant, RELATIONS)
