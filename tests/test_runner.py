from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from alexein_bench.core import Json, Question, Unit
from alexein_bench.memory import leftovers, user_id
from alexein_bench.models import Models
from alexein_bench.runner import Runner, Settings, plan_resume, steps, summarize, trimmed
from tests.fakes import FakeAlexein, FakeModels


def item(n: int, at: str) -> Json:
    return {"id": f"i{n}", "text": f"said {n}", "kind": "message", "at": at}


def q(n: int, cutoff: str | None) -> Question:
    return Question(id=f"q{n}", question=f"what {n}?", answer="x", category="c", cutoff=cutoff)


class EchoBench:
    """Scores 1 when the context holds the answer's words."""

    name = "echo"
    title = "Echo"
    variants: tuple[str, ...] = ("all",)
    models: tuple[str, ...] = ("openai/gpt-4.1-mini",)
    tokens_per_question = 100
    context_chars = 10_000

    def fetch(self, cache: Path) -> Path:
        return cache

    def units(self, data: Path, variant: str) -> Any:
        return iter([])

    def ask(self, models: Models, question: Question, context: str) -> Json:
        predicted = models.chat("openai/gpt-4.1-mini", [{"role": "user", "content": context}])
        return {"predicted": predicted, "score": float(question.answer in context)}

    def summary(self, rows: list[Json], variant: str) -> dict[str, str]:
        return {"all": f"{100 * sum(r.get('score', 0) for r in rows) / len(rows):.1f}"}


def test_steps_cut_strictly_before_each_question() -> None:
    unit = Unit(
        "u",
        [item(3, "2024-01-03T09:00:00"), item(1, "2024-01-01T09:00:00"), item(2, "2024-01-02")],
        [q(2, "2024-01-03"), q(1, "2024-01-02"), q(3, None), q(4, "2024-01-02")],
    )
    got = [([i["id"] for i in items], [x.id for x in qs]) for items, qs in steps(unit)]
    assert got == [(["i1"], ["q1", "q4"]), (["i2"], ["q2"]), (["i3"], ["q3"])]


def test_steps_with_no_dates_ask_after_everything() -> None:
    unit = Unit("u", [item(1, "2024-01-01"), item(2, "2024-01-02")], [q(1, None)])
    assert [([i["id"] for i in a], [x.id for x in b]) for a, b in steps(unit)] == [
        (["i1", "i2"], ["q1"])
    ]


def test_trimmed_takes_whole_units_first() -> None:
    units = [Unit(f"u{n}", [], [q(k, None) for k in range(3)]) for n in range(4)]
    kept = trimmed(units, limit=5, unit_limit=None)
    assert [len(u.questions) for u in kept] == [3, 2]
    assert len(trimmed(units, limit=None, unit_limit=2)) == 2


def test_user_id_fits_the_api() -> None:
    name = user_id("rhelm", "Justice_Arvind_Prakash_Mehra", "20261117a1b2c3")
    assert len(name) <= 35
    assert name.startswith("bench-20261117a1b2c3-")
    assert name != user_id("rhelm", "Elena_Morales", "20261117a1b2c3")


@pytest.fixture(autouse=True)
def no_poll_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("alexein_bench.memory._POLL", 0.0)


def run(tmp_path: Path, fake: FakeAlexein, units: list[Unit]) -> tuple[Runner, FakeModels]:
    models = FakeModels(lambda _m, _p: "ok")
    settings = Settings(bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=2)
    runner = Runner(settings, fake.client(), cast(Models, models))
    runner.run(units, lambda _line: None)
    return runner, models


def test_a_question_sees_only_what_came_before_it(tmp_path: Path) -> None:
    fake = FakeAlexein(slow=2)
    unit = Unit(
        "u",
        [item(1, "2024-01-01T10:00:00"), item(2, "2024-01-02T10:00:00")],
        [
            Question("q1", "first?", "said 1", "c", cutoff="2024-01-02"),
            Question("q2", "second?", "said 2", "c", cutoff="2024-01-02"),
            Question("q3", "last?", "said 2", "c", cutoff=None),
        ],
    )
    runner, _ = run(tmp_path, fake, [unit])
    seen = {query: ids for _, query, ids in fake.searches}
    assert seen == {"first?": ["i1"], "second?": ["i1"], "last?": ["i1", "i2"]}
    scores = {r["id"]: r["score"] for r in runner.rows()}
    assert scores == {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    assert runner.added == 2
    assert runner.searches == 3


def test_users_are_erased_and_rows_written(tmp_path: Path) -> None:
    fake = FakeAlexein()
    units = [Unit(f"u{n}", [item(n, "2024-01-01")], [q(n, None)]) for n in range(3)]
    runner, _ = run(tmp_path, fake, units)
    assert not fake.items
    # Erased once each, when done: a fresh run's users are new.
    assert len(fake.deleted) == 3
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["q0", "q1", "q2"]
    assert summarize(EchoBench(), "all", runner.rows())["answered"] == 3


def test_resume_asks_only_what_is_missing(tmp_path: Path) -> None:
    fake = FakeAlexein()
    units = [Unit("u", [item(1, "2024-01-01")], [q(1, None), q(2, None)])]
    (tmp_path / "results.jsonl").write_text(json.dumps({"id": "q1", "score": 1.0}) + "\n")
    runner, models = run(tmp_path, fake, units)
    assert len(models.calls) == 1
    assert sorted(r["id"] for r in runner.rows()) == ["q1", "q2"]


def test_a_failed_question_is_a_row(tmp_path: Path) -> None:
    fake = FakeAlexein()
    settings = Settings(bench=EchoBench(), variant="all", run="r1", out=tmp_path)

    def boom(_m: str, _p: str) -> str:
        raise RuntimeError("model down")

    runner = Runner(settings, fake.client(), cast(Models, FakeModels(boom)))
    runner.run([Unit("u", [item(1, "2024-01-01")], [q(1, None)])], lambda _line: None)
    (row,) = runner.rows()
    assert "model down" in row["error"]


def test_leftovers_finds_one_runs_users() -> None:
    fake = FakeAlexein()
    memory = fake.client()
    for name in ("bench-r1-aaaa", "bench-r2-bbbb", "someone"):
        memory.user(name).add({"id": "x", "text": "t", "at": "2024-01-01"})
    assert leftovers(memory, "r1") == ["bench-r1-aaaa"]
    assert sorted(leftovers(memory)) == ["bench-r1-aaaa", "bench-r2-bbbb"]


def test_batches_stay_under_the_request_limit() -> None:
    from alexein_bench.memory import batches

    big = [{"id": f"b{n}", "text": "x" * 300_000, "at": "2024-01-01"} for n in range(7)]
    small = [{"id": f"s{n}", "text": "y", "at": "2024-01-01"} for n in range(1200)]
    assert [len(b) for b in batches(big)] == [2, 2, 2, 1]
    assert [len(b) for b in batches(small)] == [500, 500, 200]
    assert [i["id"] for b in batches(big + small) for i in b] == [i["id"] for i in big + small]


def test_items_after_the_last_question_are_not_added() -> None:
    unit = Unit("u", [item(1, "2024-01-01"), item(2, "2024-01-05")], [q(1, "2024-01-02")])
    assert [[i["id"] for i in a] for a, _ in steps(unit)] == [["i1"]]


def test_every_question_leaves_a_trace(tmp_path: Path) -> None:
    fake = FakeAlexein()
    run(tmp_path, fake, [Unit("u", [item(1, "2024-01-01")], [q(1, None)])])
    (trace,) = [json.loads(x) for x in (tmp_path / "traces.jsonl").read_text().splitlines()]
    assert trace["query"] == "what 1?"
    assert trace["context"] == "- 2024-01-01: said 1"
    assert trace["search"]["memories"][0]["text"] == "said 1"
    assert trace["calls"][0]["reply"] == "ok"
    assert trace["row"]["score"] == 0.0


def test_the_score_so_far_every_n_questions(tmp_path: Path) -> None:
    fake = FakeAlexein()
    lines: list[str] = []
    settings = Settings(
        bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=1, report_every=3
    )
    runner = Runner(settings, fake.client(), cast(Models, FakeModels(lambda _m, _p: "ok")))
    unit = Unit("u", [item(1, "2024-01-01")], [q(n, None) for n in range(7)])
    runner.run([unit], lines.append)
    running = (tmp_path / "running.txt").read_text().splitlines()
    # After 3, after 6, and at the end.
    assert len(running) == 3
    assert "3 answered" in running[0]
    assert "7 answered" in running[-1]
    assert json.loads((tmp_path / "summary.json").read_text())["questions"] == 7


def test_plan_resume_picks_up_or_rebuilds() -> None:
    plan = [([item(1, "a")], [q(1, "b")]), ([item(2, "b")], [q(2, "c")]), ([item(3, "c")], [])]
    assert plan_resume(plan, {"i1"}, {"q1"}) == 1
    assert plan_resume(plan, {"i1", "i2"}, {"q1"}) == 1
    # Holds an item q2 must not see: build again.
    assert plan_resume(plan, {"i1", "i2", "i3"}, {"q1"}) is None
    assert plan_resume(plan, set(), set()) == 0


def test_resume_carries_on_in_the_same_memory(tmp_path: Path) -> None:
    fake = FakeAlexein()
    unit = Unit(
        "u",
        [item(1, "2024-01-01T10:00:00"), item(2, "2024-01-02T10:00:00")],
        [
            Question("q1", "first?", "said 1", "c", cutoff="2024-01-02"),
            Question("q2", "last?", "said 2", "c", cutoff=None),
        ],
    )
    calls = {"n": 0}

    def flaky(_m: str, _p: str) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return "ok"

    settings = Settings(bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=1)
    stopped = Runner(settings, fake.client(), cast(Models, FakeModels(flaky)))
    with pytest.raises(KeyboardInterrupt):
        stopped.run([unit], lambda _line: None)
    name = user_id("echo", "u", "r1")
    assert [i["id"] for i in fake.items[name]] == ["i1", "i2"]  # kept for --resume

    resumed = Settings(
        bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=1, resume=True
    )
    models = FakeModels(lambda _m, _p: "ok")
    runner = Runner(resumed, fake.client(), cast(Models, models))
    runner.run([unit], lambda _line: None)
    assert len(models.calls) == 1  # only q2
    assert runner.added == 0  # nothing added again
    assert sorted(r["id"] for r in runner.rows()) == ["q1", "q2"]
    assert name in fake.deleted  # erased once done


def test_resume_rebuilds_a_memory_that_holds_too_much(tmp_path: Path) -> None:
    fake = FakeAlexein()
    name = user_id("echo", "u", "r1")
    fake.client().user(name).add([item(1, "2024-01-01"), item(2, "2024-01-03")])
    unit = Unit(
        "u",
        [item(1, "2024-01-01"), item(2, "2024-01-03")],
        [Question("q1", "first?", "said 1", "c", cutoff="2024-01-02"), q(2, None)],
    )
    settings = Settings(
        bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=1, resume=True
    )
    runner = Runner(settings, fake.client(), cast(Models, FakeModels(lambda _m, _p: "ok")))
    runner.run([unit], lambda _line: None)
    seen = {query: ids for _, query, ids in fake.searches}
    assert seen["first?"] == ["i1"]
    assert runner.added == 2


def test_stopping_finishes_whats_in_flight_and_starts_nothing(tmp_path: Path) -> None:
    fake = FakeAlexein()
    units = [Unit(f"u{n}", [item(n, "2024-01-01")], [q(n * 10 + k, None) for k in range(3)])
             for n in range(4)]  # fmt: skip
    calls = {"n": 0}
    settings = Settings(bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=2)

    def stop_on_second(_m: str, _p: str) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            runner._stop.set()  # what Ctrl-C in the main thread does
        return "ok"

    runner = Runner(settings, fake.client(), cast(Models, FakeModels(stop_on_second)))
    runner.run(units, lambda _line: None)
    # The question in flight on the other unit may finish; nothing new starts.
    assert calls["n"] <= 3
    assert len(runner.rows()) == calls["n"]
    assert len(fake.items) <= 2  # started units kept for --resume, the rest never added
    assert not fake.deleted
    assert (tmp_path / "summary.json").exists()


def test_an_outage_is_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from alexein import ServiceUnavailableError

    from alexein_bench.memory import patient

    monkeypatch.setattr("alexein_bench.memory._OUTAGE_WAIT", 0.0)
    tries = {"n": 0}

    def flaky() -> str:
        tries["n"] += 1
        if tries["n"] < 3:
            raise ServiceUnavailableError("502", status=502, detail="<!DOCTYPE html>...")
        return "ok"

    assert patient(flaky) == "ok"
    assert tries["n"] == 3


def test_a_wrong_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    from alexein import InvalidRequestError

    from alexein_bench.memory import patient

    monkeypatch.setattr("alexein_bench.memory._OUTAGE_WAIT", 0.0)
    tries = {"n": 0}

    def wrong() -> str:
        tries["n"] += 1
        raise InvalidRequestError("422", status=422, detail="bad field")

    with pytest.raises(InvalidRequestError):
        patient(wrong)
    assert tries["n"] == 1


def test_an_error_is_one_line() -> None:
    from alexein import ServiceUnavailableError

    from alexein_bench.memory import short

    page = ServiceUnavailableError("502", status=502, detail="<!DOCTYPE html><html>...")
    assert short(page) == "Alexein answered 502"
    assert short(ValueError("x")) == "ValueError: x"


def test_resume_of_a_unit_never_started(tmp_path: Path) -> None:
    fake = FakeAlexein()
    settings = Settings(
        bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=1, resume=True
    )
    runner = Runner(settings, fake.client(), cast(Models, FakeModels(lambda _m, _p: "ok")))
    runner.run([Unit("u", [item(1, "2024-01-01")], [q(1, None)])], lambda _line: None)
    (row,) = runner.rows()
    assert "error" not in row


def test_a_long_wait_says_how_far_it_has_got(monkeypatch: pytest.MonkeyPatch) -> None:
    from alexein_bench.memory import Account

    monkeypatch.setattr("alexein_bench.memory._REPORT", 0.0)
    fake = FakeAlexein(slow=2)
    account = Account(fake.client(), "u1", timeout=60)
    ids = account.add([item(1, "2024-01-01"), item(2, "2024-01-02")])
    heard: list[tuple[int, int]] = []
    account.wait(ids, report=lambda done, of, _s: heard.append((done, of)))
    assert heard and heard[0] == (0, 2)


def test_a_status_line_every_few_seconds(tmp_path: Path) -> None:
    fake = FakeAlexein(slow=40)  # each item takes a while to be processed
    lines: list[str] = []
    settings = Settings(
        bench=EchoBench(), variant="all", run="r1", out=tmp_path, workers=2, heartbeat=0.02
    )
    runner = Runner(settings, fake.client(), cast(Models, FakeModels(lambda _m, _p: "ok")))
    units = [Unit(f"u{n}", [item(n, "2024-01-01")], [q(n, None)]) for n in range(3)]
    from alexein_bench import memory

    old, memory._POLL = memory._POLL, 0.005
    try:
        runner.run(units, lines.append)
    finally:
        memory._POLL = old
    beats = [x for x in lines if "answered" in x and "units" in x]
    assert beats, lines
    assert any("processing 0/1 items" in x for x in beats)
    assert "/3 answered" in beats[0]
    assert runner.status().startswith(("0s", "1s"))
    assert "units 3 done, 0 running, 0 waiting" in runner.status()


def test_settle_is_asked_once_everything_is_stored(tmp_path: Path) -> None:
    fake = FakeAlexein(slow=50, settles=True)
    unit = Unit("u", [item(1, "2024-01-01"), item(2, "2024-01-02")], [q(1, None)])
    runner, _ = run(tmp_path, fake, [unit])
    ((_, stored),) = fake.settled
    assert sorted(stored) == ["i1", "i2"], "asked only once both were stored"
    assert "error" not in runner.rows()[0]


def test_without_settle_it_waits_at_memorys_pace(tmp_path: Path) -> None:
    fake = FakeAlexein(slow=3)
    runner, _ = run(tmp_path, fake, [Unit("u", [item(1, "2024-01-01")], [q(1, None)])])
    assert not fake.settled
    assert "error" not in runner.rows()[0]
