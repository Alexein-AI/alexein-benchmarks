"""BEAM's loader, grading and headline numbers, offline: a fixture conversation and a fake
model, no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alexein_bench.benches import beam
from alexein_bench.benches import beam_prompts as prompts
from alexein_bench.core import Question

CONVERSATION: dict[str, Any] = {
    "conversation_id": "7",
    "chat": [
        [
            {
                "role": "user",
                "content": "I start at Acme on Monday.",
                "time_anchor": "March-01-2024",
            },
            {"role": "assistant", "content": "Congratulations!", "time_anchor": ""},
            {"role": "user", "content": "   ", "time_anchor": ""},
        ],
        [
            {"role": "human", "content": "Moved to Lisbon.", "time_anchor": ""},
            {"role": "assistant", "content": "Noted.", "time_anchor": ""},
        ],
    ],
    "probing_questions": repr(
        {
            "information_extraction": [
                {
                    "question": "Where do I work?",
                    "rubric": ["Acme", "Monday"],
                    "difficulty": "easy",
                },
            ],
            "abstention": [{"question": "What is my cat called?", "rubric": []}],
            "unlisted_ability": [{"question": "ignored", "rubric": ["x"]}],
        }
    ),
}


class FakeModels:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> str:
        self.calls.append({"model": model, "messages": messages, **kw})
        return self.replies.pop(0)


def test_items_dated_by_batch_anchor_and_ordered() -> None:
    items = beam.items_of(CONVERSATION)
    assert [i["text"] for i in items] == [
        "I start at Acme on Monday.",
        "Congratulations!",
        "Moved to Lisbon.",
        "Noted.",
    ]
    assert items[0]["at"] == "2024-03-01T00:00:00"
    assert items[1]["at"] == "2024-03-01T00:01:00"
    # A batch with no anchor is the day after the one before.
    assert items[2]["at"] == "2024-03-02T00:00:00"
    assert items[2]["role"] == "user"
    assert items[2]["session"] == "batch-2"
    assert len({i["id"] for i in items}) == len(items)


def test_questions_in_ability_order_with_rubric() -> None:
    questions = beam.questions_of(CONVERSATION)
    assert [q.category for q in questions] == ["abstention", "information_extraction"]
    q = questions[1]
    assert q.question == "Where do I work?"
    assert q.answer == "Acme | Monday"
    assert q.extra["rubric"] == ["Acme", "Monday"]
    assert q.cutoff is None


def test_grade_reads_as_theirs() -> None:
    assert beam.grade('{"score": 0.8, "reason": "x"}') == 1.0
    assert beam.grade('{"score": 0.3}') == 0.5
    assert beam.grade('{"score": 0.1}') == 0.0
    assert beam.grade('{"final": "{\\"score\\": 1.0}"}') == 1.0
    assert beam.grade('{"score": "high"}') == 0.0
    assert beam.grade("[1.0]") == 1.0
    assert beam.grade("not json") is None


def test_ask_scores_rubric_mean_and_retries() -> None:
    question = Question(
        id="7/information_extraction/0",
        question="Where do I work?",
        answer="Acme | Monday",
        category="information_extraction",
        extra={"rubric": ["Acme", "Monday"]},
    )
    models = FakeModels(
        ["", "Thinking...\nANSWER: Acme, from Monday", "oops", '{"score": 1.0}', '{"score": 0.5}']
    )
    row = beam.Beam().ask(models, question, "")  # type: ignore[arg-type]
    assert row["predicted"] == "Acme, from Monday"
    assert row["nuggets"] == [1.0, 0.5]
    assert row["score"] == 0.75
    answer_call = models.calls[0]
    assert answer_call["model"] == "openai/gpt-5"
    assert answer_call["temperature"] is None
    assert answer_call["max_tokens"] == 4096
    assert answer_call["messages"] == [
        {
            "role": "user",
            "content": prompts.ANSWER_PROMPT.format(
                question="Where do I work?", memories_text=prompts.NO_MEMORIES
            ),
        }
    ]
    judge_call = models.calls[2]
    assert judge_call["json_object"] is True
    assert judge_call["messages"][0] == {"role": "system", "content": prompts.JUDGE_SYSTEM}
    assert "RUBRIC CRITERION:\nAcme\n" in judge_call["messages"][1]["content"]


def test_no_rubric_scores_zero() -> None:
    question = Question(id="q", question="?", answer="", category="abstention")
    row = beam.Beam().ask(FakeModels(["An answer."]), question, "ctx")  # type: ignore[arg-type]
    assert row["score"] == 0.0


def test_summary_is_written_as_the_website_writes_it() -> None:
    rows: list[dict[str, Any]] = [{"score": 1.0}, {"score": 0.5}, {"score": 0.25}, {"error": "x"}]
    got = beam.Beam().summary(rows, "1m")
    assert got["mean"] == "43.75"
    assert got["pass"] == "50.00%"


def test_units_from_converted_json(tmp_path: Path) -> None:
    data = tmp_path / "1M.json"
    data.write_text(json.dumps([CONVERSATION]))
    units = list(beam.Beam().units(data, "1m"))
    assert [u.id for u in units] == ["1m-7"]
    assert len(units[0].items) == 4
    assert len(units[0].questions) == 2
