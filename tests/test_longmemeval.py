from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from alexein_bench.benches import longmemeval_prompts as prompts
from alexein_bench.benches.longmemeval import EMPTY, JUDGE, READER, LongMemEval, load
from alexein_bench.models import Models
from alexein_bench.runner import steps
from tests.fakes import FakeModels


def raw(qid: str = "q1", qtype: str = "temporal-reasoning") -> dict[str, object]:
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": "How many days ago did I buy the bike?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "10 days",
        "answer_session_ids": ["s2"],
        "haystack_session_ids": ["s1", "s2"],
        # The second session is dated after the question, as 76 of the dataset's are.
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/06/02 (Fri) 10:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            [{"role": "user", "content": "I bought a bike"}, {"role": "assistant", "content": ""}],
        ],
    }


def test_load_places_turns_and_names_speakers() -> None:
    unit = load(raw())
    assert [i["id"] for i in unit.items] == ["q1:0:0", "q1:0:1", "q1:1:0"]
    assert [i["at"] for i in unit.items] == [
        "2023-05-20T02:21:00",
        "2023-05-20T02:21:30",
        "2023-06-02T10:00:00",
    ]
    assert unit.items[1]["author"] == "assistant_q1"
    assert unit.items[2]["session"] == "session_1"
    (question,) = unit.questions
    assert question.date == "2023/05/30 (Tue) 23:40"


def test_question_sees_the_whole_haystack() -> None:
    (only,) = steps(load(raw()))
    items, questions = only
    assert len(items) == 3
    assert [q.id for q in questions] == ["q1"]


def test_ask_answers_then_judges_with_type_clause() -> None:
    def reply(model: str, prompt: str) -> str:
        if model == READER:
            return "reasoning...\n## STEP 7: FINAL ANSWER\n10 days"
        return 'ok {"label": "CORRECT"}'

    fake = FakeModels(reply)
    question = load(raw()).questions[0]
    row = LongMemEval().ask(cast(Models, fake), question, "[2023-06-02] I bought a bike")
    assert row == {"predicted": "10 days", "verdict": "CORRECT", "score": 1.0}
    (reader, reader_prompt), (judge, judge_prompt) = fake.calls
    assert (reader, judge) == (READER, JUDGE)
    assert "Current Date: 2023/05/30 (Tue) 23:40\n" in reader_prompt
    assert "between user_q1 and assistant_q1" in reader_prompt
    assert judge_prompt.startswith(prompts._CLAUSES["temporal-reasoning"])


def test_abstention_clause_wins_over_type() -> None:
    assert prompts.leniency_clause("x_abs", "temporal-reasoning") == prompts._ABSTAIN_CLAUSE
    assert prompts.leniency_clause("x", "multi-session") == ""


def test_empty_context_is_wrong_without_a_call() -> None:
    fake = FakeModels(lambda *_: "")
    row = LongMemEval().ask(cast(Models, fake), load(raw()).questions[0], "  ")
    assert row == {"predicted": EMPTY, "verdict": "WRONG", "score": 0.0}
    assert fake.calls == []


def test_unreadable_judge_is_asked_four_times_then_left_out() -> None:
    fake = FakeModels(lambda model, _: "FINAL ANSWER: x" if model == READER else "no json")
    row = LongMemEval().ask(cast(Models, fake), load(raw()).questions[0], "ctx")
    assert row == {"predicted": "x", "verdict": "UNJUDGED"}
    assert len(fake.calls) == 5


def test_parse_judge_label() -> None:
    assert prompts.parse_judge_label('{"label": "wrong"}') == "WRONG"
    assert prompts.parse_judge_label('{"label": "maybe"}') is None
    assert prompts.parse_judge_label("CORRECT") is None
    assert prompts.extract_final_answer("a FINAL ANSWER: b") == "b"


def test_summary_counts_errors_wrong_and_leaves_unjudged_out() -> None:
    rows = [
        {"category": "multi-session", "score": 1.0},
        {"category": "multi-session", "score": 0.0},
        {"category": "knowledge-update", "error": "boom"},
        {"category": "knowledge-update", "verdict": "UNJUDGED"},
    ]
    got = LongMemEval().summary(rows, "all")
    assert got["all"] == "33.3"
    assert got["all:multi-session"] == "50.0"
    assert got["all:knowledge-update"] == "0.0"
    assert got["all:unjudged"] == "1"


def test_units_read_the_file(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps([raw("a"), raw("b_abs", "multi-session")]))
    units = list(LongMemEval().units(path, "all"))
    assert [u.id for u in units] == ["a", "b_abs"]
    assert units[1].questions[0].extra == {"abstention": True}
