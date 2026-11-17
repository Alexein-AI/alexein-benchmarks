"""LoCoMo and LoCoMo-Plus, offline: loaders on a small fixture, placement, label reading and
scoring, and the ask path with a fake model."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from alexein_bench.benches import locomo, locomo_plus
from alexein_bench.core import Question
from alexein_bench.runner import steps

C = locomo_plus.FULL_COLON

CONV = {
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hi Mel!"},
            {
                "speaker": "Melanie",
                "dia_id": "D1:2",
                "text": "Look at this.",
                "blip_caption": "a photo of a dog",
            },
        ],
        "session_2_date_time": "10:00 am on 20 June, 2023",
        "session_2": [{"speaker": "Caroline", "dia_id": "D2:1", "text": "I adopted a cat."}],
    },
    "qa": [
        {"question": "What did Mel share?", "answer": "a dog photo", "category": 4},
        {"question": "When?", "answer": "May 2023", "category": 2},
        {"question": "Trick?", "adversarial_answer": "none", "category": 5},
    ],
}


def _write(tmp_path: Path, name: str, data: Any) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


class FakeModels:
    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> str:
        self.calls.append({"model": model, "messages": messages, **kw})
        return self.replies[model].pop(0)


def test_parse_when() -> None:
    assert locomo.parse_when("1:56 pm on 8 May, 2023") == dt.datetime(
        2023, 5, 8, 13, 56, tzinfo=dt.UTC
    )
    assert locomo.parse_when("12:05 am on 1 January 2024") == dt.datetime(
        2024, 1, 1, 0, 5, tzinfo=dt.UTC
    )
    assert locomo.parse_when("someday") is None


def test_units(tmp_path: Path) -> None:
    path = _write(tmp_path, "locomo10.json", [CONV])
    (unit,) = list(locomo.Locomo().units(path, "all"))
    assert unit.id == "conv-1"
    assert [i["at"] for i in unit.items] == [
        "2023-05-08T13:56:00Z",
        "2023-05-08T13:56:45Z",
        "2023-06-20T10:00:00Z",
    ]
    assert unit.items[1]["text"] == "Look at this. [shares a photo: a photo of a dog]"
    assert unit.items[0]["author"] == {"id": "conv-1/Caroline", "name": "Caroline", "self": True}
    assert unit.items[1]["author"]["self"] is False
    assert unit.items[2]["session"] == "conv-1/session_2"
    # Category 5 is left out; the rest are asked after everything.
    assert [q.category for q in unit.questions] == ["single-hop", "temporal"]
    assert all(q.cutoff is None for q in unit.questions)
    ((added, asked),) = steps(unit)
    assert len(added) == 3 and len(asked) == 2


def test_final_answer_and_label() -> None:
    assert locomo.final_answer("x\n## STEP 7: FINAL ANSWER\nA dog photo.") == "A dog photo."
    assert locomo.final_answer("FINAL ANSWER: one FINAL ANSWER: two") == "two"
    assert locomo.final_answer("plain") == "plain"
    assert locomo.label('Because.\n```json\n{"label": "CORRECT"}\n```') == "CORRECT"
    assert locomo.label('ok {"label": "wrong"}') == "WRONG"
    with pytest.raises(ValueError, match="unknown"):
        locomo.label('{"label": "maybe"}')


def test_locomo_ask_majority() -> None:
    models = FakeModels(
        {
            locomo.ANSWER_MODEL: ["...\n## STEP 7: FINAL ANSWER\nA dog photo"],
            locomo.JUDGE_MODEL: [
                '{"label": "CORRECT"}',
                '{"label": "WRONG"}',
                "junk",
                '{"label": "CORRECT"}',
            ],
        }
    )
    q = Question(
        "q",
        "What did Mel share?",
        "a dog photo",
        "single-hop",
        extra={"speaker_a": "Caroline", "speaker_b": "Melanie"},
    )
    row = locomo.Locomo().ask(models, q, "Melanie shared a photo of a dog.")  # type: ignore[arg-type]
    assert row["predicted"] == "A dog photo"
    assert row["votes"] == [True, False, True]
    assert row["score"] == 1.0
    assert (
        "Episodes memories for conversation between Caroline and Melanie"
        in (models.calls[0]["messages"][0]["content"])
    )
    assert models.calls[1]["messages"][0]["role"] == "system"
    # Each of the three asks is its own; the unreadable reply is asked again.
    assert len({c.get("seed") for c in models.calls[1:]}) == 4


def test_empty_context_is_wrong() -> None:
    q = Question("q", "x", "y", "temporal", extra={"speaker_a": "A", "speaker_b": "B"})
    row = locomo.Locomo().ask(FakeModels({}), q, "  ")  # type: ignore[arg-type]
    assert row["score"] == 0.0


def test_summary_leaves_errors_out() -> None:
    rows = [
        {"category": "temporal", "score": 1.0},
        {"category": "single-hop", "score": 0.0},
        {"category": "temporal", "error": "boom"},
    ]
    got = locomo.Locomo().summary(rows, "all")
    assert got == {"all": "50.0", "all:temporal": "100.0", "all:single-hop": "0.0"}


PLUS = [
    {
        "relation_type": "causal",
        "cue_dialogue": "A: I hurt my knee running.\nB: Oh no!",
        "trigger_query": "B: Want to go hiking this weekend?",
        "time_gap": "two weeks later",
    },
    {
        "relation_type": "value",
        "cue_dialogue": "A: I never eat meat.",
        "trigger_query": "B: Let's get burgers.",
        "time_gap": "several months later",
    },
]


def test_gap_days() -> None:
    assert locomo_plus.gap_days("two weeks later") == 14
    assert locomo_plus.gap_days("3 months later") == 90
    assert locomo_plus.gap_days("a year later") == 365
    assert locomo_plus.gap_days("several months later") == 0


def test_plus_units(tmp_path: Path) -> None:
    """One app user per conversation, holding every cue placed in it."""
    _write(tmp_path, "locomo10.json", [CONV])
    _write(tmp_path, "locomo_plus.json", PLUS)
    (unit,) = list(locomo_plus.LocomoPlus().units(tmp_path, "all"))
    # Asked a week after the last session (20 June); the first cue two weeks before that:
    # 13 June, between the two sessions.
    first = [i for i in unit.items if "/C0:" in i["id"]]
    assert [c["text"] for c in first] == ["I hurt my knee running.", "Oh no!"]
    assert first[0]["at"] == "2023-06-13T10:00:00Z"
    assert first[0]["author"]["name"] == "Caroline" and first[1]["author"]["name"] == "Melanie"
    # An unreadable gap puts the second cue at the trigger's own time, after every session.
    (second,) = [i for i in unit.items if "/C1:" in i["id"]]
    assert second["at"] == "2023-06-27T10:00:00Z"
    assert len(unit.items) == 6
    assert [i["at"] for i in unit.items] == sorted(i["at"] for i in unit.items)
    assert len({i["id"] for i in unit.items}) == 6, "no two cues share an id"
    hike, burgers = unit.questions
    assert (hike.id, burgers.id) == ("plus-0", "plus-1")
    assert hike.question == "Want to go hiking this weekend?"
    assert hike.answer == f"Caroline{C}I hurt my knee running.\nMelanie{C}Oh no!"
    assert hike.category == "causal" and hike.cutoff is None
    assert burgers.category == "value"


def test_plus_label_quirks() -> None:
    assert locomo_plus.label('{"label": "correct", "reason": "yes"}') == "correct"
    assert locomo_plus.label('{"label": "wrong"}') == "wrong"
    assert locomo_plus.label("This is incorrect.") == "correct"
    assert locomo_plus.label("nothing") == ""


def test_plus_ask(tmp_path: Path) -> None:
    _write(tmp_path, "locomo10.json", [CONV])
    _write(tmp_path, "locomo_plus.json", PLUS)
    unit = next(iter(locomo_plus.LocomoPlus().units(tmp_path, "all")))
    models = FakeModels(
        {
            locomo_plus.READER_MODEL: ["  "],
            locomo_plus.JUDGE_MODEL: ['{"label": "partial", "reason": "some"}'],
        }
    )
    row = locomo_plus.LocomoPlus().ask(models, unit.questions[0], "memory")  # type: ignore[arg-type]
    assert row == {"predicted": "(empty)", "label": "partial", "score": 0.5}
    reader = models.calls[0]["messages"][0]["content"]
    assert reader.startswith("Below is a conversation between two people: Caroline and Melanie.")
    assert reader.endswith('memory\n\nMelanie said, "Want to go hiking this weekend?"')
    judge = models.calls[1]["messages"][0]["content"]
    assert judge.startswith("Below is a conversation between two people: A and B.")
    assert f"Caroline{C}I hurt my knee running." in judge
    assert models.calls[0]["temperature"] == 0.3
