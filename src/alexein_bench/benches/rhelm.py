"""RHELM (Microsoft, arXiv 2605.31086): a year and more of ten people's lives - daily chats
with an assistant, their emails and the documents they worked on - and 1,305 questions.

Its leaderboard has two tables, the two variants here: `without` holds the chats only, `with`
the emails and attachments too. Every question is asked in both.

How a run goes, as RHELM scores it:

- A character is one app user. Each chat day is a session, each turn its user and assistant
  messages at the turn's own time. An email file is a day's emails split on `====` lines, each
  a document titled by its subject, at noon of its day. An attachment is a document, its html
  read as its text, dated by the email with the same leading sequence number (or, failing
  that, the conversation with that number).
- A question is asked on its `question_date` and sees only what came strictly before that day:
  items are added a question date at a time, and each question waits until memory has
  processed everything before it.
- Reader and judge are gpt-4.1-mini, sampled as RHELM's own client samples them: temperature
  0.7, no seed, no reply read back from a cache. A run is one draw, as each leaderboard row is.
- The score is RHELM's: accuracy 0 or 1, averaged over every question; a question that fails
  scores 0.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from alexein_bench.benches import rhelm_prompts as prompts
from alexein_bench.core import Json, Question, Unit
from alexein_bench.fetch import clone
from alexein_bench.models import Models

REPO = "https://github.com/microsoft/RHELM"
COMMIT = "5e170da7a2db96b4d222620a2aaaddb031cb8098"
MODEL = "openai/gpt-4.1-mini"
TEMPERATURE = 0.7
# RHELM's: an accuracy that is not 0 or 1 counts as 1 from a half up.
_PASS = 0.5
CATEGORIES = (
    "fact",
    "temporal",
    "aggregation",
    "hallucination",
    "misleading",
    "attachment",
    "mixed",
)
# The ten, smallest first.
CHARACTERS = (
    "Elena_Morales",
    "Dr._Alastair_McEwan",
    "Jonas_Richter",
    "Dr._Elena_Markovic",
    "Dr._Elias_Romero",
    "Justice_Arvind_Prakash_Mehra",
    "David_Reyes",
    "Dr._Marcus_Ellison",
    "David_R._Ellis",
    "Martin_Keller",
)
_SEQ = re.compile(r"^(\d+)_")
_EMAIL_DAY = re.compile(r"_(\d{4})_(\d{2})_(\d{2})\.txt$")


class _Text(HTMLParser):
    """Visible text of an html report: everything but its style and script blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:  # noqa: ARG002 - HTMLParser's
        self._hidden += tag in ("style", "script")

    def handle_endtag(self, tag: str) -> None:
        self._hidden -= tag in ("style", "script") and self._hidden > 0

    def handle_data(self, data: str) -> None:
        if not self._hidden and data.strip():
            self.parts.append(data.strip())


def _html_text(raw: str) -> str:
    parser = _Text()
    parser.feed(raw)
    return "\n".join(parser.parts)


def _conversations(name: str, root: Path) -> list[tuple[str, list[Json]]]:
    """Each day's date and its items, in date order."""
    out = []
    for path in sorted((root / "conversations" / name).glob("conversation_*.json")):
        raw = json.loads(path.read_text())
        day, items = str(raw["date"]), []
        for turn in raw["conversation"]:
            cite = f"{day}:{turn['turn']}"
            for role in ("user", "assistant"):
                if not str(turn.get(role) or "").strip():
                    continue
                items.append(
                    {
                        "id": f"{name}/{cite}/{role}",
                        "kind": "message",
                        "session": day,
                        "role": role,
                        "author": name.replace("_", " ") if role == "user" else "Assistant",
                        "text": turn[role],
                        "at": turn.get("timestamp") or f"{day}T12:00:00",
                    }
                )
        out.append((day, items))
    return out


def _emails(name: str, root: Path) -> tuple[list[Json], dict[int, str]]:
    """Every email as a document, and each email file's sequence number -> its day."""
    items, days = [], {}
    for path in sorted((root / "emails" / name).glob("*.txt")):
        seq, found = _SEQ.match(path.name), _EMAIL_DAY.search(path.name)
        if not (seq and found):
            continue
        day = "-".join(found.groups())
        days[int(seq.group(1))] = day
        blocks = [b.strip() for b in re.split(r"\n=+\n", path.read_text()) if b.strip()]
        for n, block in enumerate(blocks):
            subject = next((x[9:] for x in block.splitlines() if x.startswith("Subject: ")), "")
            items.append(
                {
                    "id": f"{name}/email/{day}/{n}",
                    "kind": "document",
                    "title": subject or f"Email of {day}",
                    "text": block,
                    "at": f"{day}T12:00:00",
                }
            )
    return items, days


def _attachments(name: str, root: Path, days: dict[int, str]) -> list[Json]:
    items = []
    for path in sorted((root / "attachments" / name).iterdir()):
        seq = _SEQ.match(path.name)
        if path.suffix not in (".md", ".html") or not seq or int(seq.group(1)) not in days:
            continue
        raw = path.read_text()
        text = _html_text(raw) if path.suffix == ".html" else raw
        if not text.strip():
            continue
        items.append(
            {
                "id": f"{name}/file/{path.name}",
                "kind": "document",
                "title": path.stem,
                "text": text,
                "at": f"{days[int(seq.group(1))]}T13:00:00",
            }
        )
    return items


def _questions(name: str, root: Path) -> list[Question]:
    path = root / "QA_final" / f"low_score_qa_{name}_all_validated.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        Question(
            id=f"{name}/{r['id']}",
            question=r["question"],
            answer=str(r["answer"]),
            category=r["question_type"],
            # Strictly before the question's day: "2024-07-24T09:00" sorts after "2024-07-24".
            cutoff=r["question_date"],
            date=r["question_date"],
            as_of=r["question_date"],
        )
        for r in rows
    ]


def load(name: str, root: Path, *, sources: bool) -> Unit:
    conversations = _conversations(name, root)
    items = [i for _, day_items in conversations for i in day_items]
    if sources:
        emails, days = _emails(name, root)
        # Theirs dates an attachment by its email's number; the n-th conversation backs that up.
        for n, (day, _) in enumerate(conversations, start=1):
            days.setdefault(n, day)
        items += emails + _attachments(name, root, days)
    return Unit(id=name, items=items, questions=_questions(name, root))


def judge_reply(text: str) -> Json:
    """Accuracy 0 or 1 and overall / 10, read as `llm_judge_score` reads them: a fence
    stripped, then `json.loads`; a score that is not 0 or 1 counts from a half up; a reply
    that does not read is 0."""
    text = text.strip().removeprefix("```json").removesuffix("```")
    try:
        got = json.loads(text)
        raw = got.get("accuracy_score", 0)
        accuracy = float(raw) if raw in (0, 1) else 1.0 if raw >= _PASS else 0.0
        return {"score": accuracy, "overall": got.get("overall_score", 0) / 10.0}
    except Exception:  # theirs: any failure to read the reply is a 0
        return {"score": 0.0, "overall": 0.0, "unread": text[:500]}


class Rhelm:
    name = "rhelm"
    title = "RHELM"
    variants: tuple[str, ...] = ("with", "without")
    models: tuple[str, ...] = (MODEL,)
    tokens_per_question = 9_000
    context_chars = 24_000

    def fetch(self, cache: Path) -> Path:
        return clone(REPO, COMMIT, cache / f"rhelm-{COMMIT[:7]}", keep=("data",)) / "data"

    def units(self, data: Path, variant: str) -> Iterator[Unit]:
        for name in CHARACTERS:
            yield load(name, data, sources=variant == "with")

    def _call(self, models: Models, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return models.chat(MODEL, messages, temperature=TEMPERATURE, reuse=False).strip()

    def ask(self, models: Models, question: Question, context: str) -> Json:
        predicted = self._call(
            models,
            prompts.ANSWER_PROMPT.format(
                context_text=context, question_date_text=question.date, query=question.question
            ),
        )
        verdict = self._call(
            models,
            prompts.JUDGE_PROMPT.format(
                question=question.question, expected=question.answer, predicted=predicted
            ),
        )
        return {"predicted": predicted, **judge_reply(verdict)}

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        mean = sum(float(r.get("score", 0.0)) for r in rows) / len(rows)
        out = {variant: f"{100 * mean:.1f}"}
        for category in CATEGORIES:
            group = [r for r in rows if r.get("category") == category]
            if group:
                share = sum(float(r.get("score", 0.0)) for r in group) / len(group)
                out[f"{variant}:{category}"] = f"{100 * share:.1f}"
        return out
