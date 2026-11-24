from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from alexein_bench.benches import rhelm_prompts as prompts
from alexein_bench.benches.rhelm import Rhelm, judge_reply, load
from alexein_bench.core import Question
from alexein_bench.models import Models
from alexein_bench.runner import steps
from tests.fakes import FakeModels

NAME = "Ada_Test"


def fixture(root: Path) -> Path:
    conv = root / "conversations" / NAME
    conv.mkdir(parents=True)
    for n, day in enumerate(("2024-02-01", "2024-02-03"), start=1):
        turns = [
            {"turn": 1, "user": f"hi {day}", "assistant": "hello", "timestamp": f"{day}T09:00:00"}
        ]
        (conv / f"conversation_{n:03}.json").write_text(
            json.dumps({"date": day, "conversation": turns})
        )
    mail = root / "emails" / NAME
    mail.mkdir(parents=True)
    (mail / "1_inbox_2024_02_01.txt").write_text(
        "Subject: Plans\nGo.\n=====\nSubject: Again\nStay."
    )
    files = root / "attachments" / NAME
    files.mkdir(parents=True)
    (files / "1_report.html").write_text("<style>x{}</style><p>Budget is 5</p>")
    (files / "2_notes.md").write_text("Second day notes")
    qa = root / "QA_final"
    qa.mkdir()
    rows = [
        {"id": "f1", "question": "Q1", "answer": "A", "question_date": "2024-02-03",
         "question_type": "fact"},
        {"id": "f2", "question": "Q2", "answer": "B", "question_date": "2024-02-04",
         "question_type": "attachment"},
    ]  # fmt: skip
    (qa / f"low_score_qa_{NAME}_all_validated.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    return root


def test_without_sources_holds_chats_only(tmp_path: Path) -> None:
    unit = load(NAME, fixture(tmp_path), sources=False)
    assert {i["kind"] for i in unit.items} == {"message"}
    assert len(unit.items) == 4
    assert [q.cutoff for q in unit.questions] == ["2024-02-03", "2024-02-04"]


def test_with_sources_adds_emails_and_attachments(tmp_path: Path) -> None:
    unit = load(NAME, fixture(tmp_path), sources=True)
    docs = {i["title"]: i for i in unit.items if i["kind"] == "document"}
    assert set(docs) == {"Plans", "Again", "1_report", "2_notes"}
    assert docs["1_report"]["text"] == "Budget is 5"
    assert docs["1_report"]["at"] == "2024-02-01T13:00:00"
    # No email numbered 2: dated by the second conversation instead.
    assert docs["2_notes"]["at"] == "2024-02-03T13:00:00"


def test_a_question_never_sees_its_own_day(tmp_path: Path) -> None:
    unit = load(NAME, fixture(tmp_path), sources=False)
    (first, asked), *_ = steps(unit)
    assert [q.id for q in asked] == [f"{NAME}/f1"]
    assert all(i["at"] < "2024-02-03" for i in first)


def test_judge_reads_replies_as_rhelm_does() -> None:
    assert judge_reply('```json\n{"accuracy_score": 1, "overall_score": 8}\n```')["score"] == 1.0
    assert judge_reply('{"accuracy_score": 0.6, "overall_score": 5}')["score"] == 1.0
    assert judge_reply('{"accuracy_score": 0.2}')["score"] == 0.0
    assert judge_reply("not json")["score"] == 0.0


def test_ask_samples_answer_then_judge() -> None:
    def reply(_model: str, prompt: str) -> str:
        return '{"accuracy_score": 1, "overall_score": 9}' if "expert evaluator" in prompt else "A"

    models = FakeModels(reply)
    question = Question("x", "Q?", "A", "fact", cutoff="2024-02-03", date="2024-02-03")
    row = Rhelm().ask(cast(Models, models), question, "- 2024-02-01: hi")
    assert row["predicted"] == "A"
    assert row["score"] == 1.0
    assert "Question Date: 2024-02-03" in models.calls[0][1]
    assert "- 2024-02-01: hi" in models.calls[0][1]


def test_summary_is_the_leaderboard_average() -> None:
    rows = [
        {"category": "fact", "score": 1.0},
        {"category": "fact", "score": 0.0},
        {"category": "mixed", "error": "boom"},
    ]
    got = Rhelm().summary(rows, "with")
    assert got["with"] == "33.3"
    assert got["with:fact"] == "50.0"


def test_a_question_is_searched_as_of_its_own_day(tmp_path: Path) -> None:
    unit = load(NAME, fixture(tmp_path), sources=False)
    assert [x.as_of for x in unit.questions] == ["2024-02-03", "2024-02-04"]


def test_judge_prompt_keeps_rhelm_s_trailing_space() -> None:
    # Verbatim, at 5e170da: `evaluation/rag_benchmark.py` ends this line with a space.
    assert "contradicts the expected answer. \n" in prompts.JUDGE_PROMPT
