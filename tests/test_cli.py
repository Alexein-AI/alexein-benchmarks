"""The one command, end to end: keys, dataset, the estimate, the run, the report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from alexein_bench import cli
from alexein_bench.benches.rhelm import Rhelm
from tests.fakes import FakeAlexein
from tests.test_rhelm import NAME, fixture


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeAlexein:
    fake = FakeAlexein()
    data = fixture(tmp_path / "data")
    monkeypatch.setattr(cli.keys, "load", lambda: ("tm_" + "k" * 43, "sk-test"))
    monkeypatch.setattr(cli, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(cli, "Memory", lambda *_a, **_k: fake.client())
    monkeypatch.setattr(Rhelm, "fetch", lambda _self, _cache: data)
    monkeypatch.setattr("alexein_bench.benches.rhelm.CHARACTERS", (NAME,))
    monkeypatch.setattr("alexein_bench.memory._POLL", 0.0)

    def answer(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        text = '{"accuracy_score": 1, "overall_score": 9}' if "expert" in prompt else "A"
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": text}}], "usage": usage}
        )

    real = cli.Models

    def models(key: str, cache: Path, **kw: Any) -> Any:
        made = real(key, cache, **kw)
        made._http = httpx.Client(transport=httpx.MockTransport(answer))
        return made

    monkeypatch.setattr(cli, "Models", models)
    return fake


def test_run_prints_the_published_number_beside_its_own(
    wired: FakeAlexein, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "runs"
    code = cli.main(["run", "rhelm", "--variant", "with", "--yes", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "1 app users" in printed
    assert "48.5" in printed
    (summary_path,) = out.glob("rhelm/with/*/summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["answered"] == 2
    assert summary["scores"]["with"] == "100.0"
    assert summary["alexein"]["searches"] == 2
    assert not wired.items  # erased


def test_run_off_a_terminal_without_yes_spends_nothing(
    wired: FakeAlexein, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["run", "rhelm", "--out", str(tmp_path / "runs")]) == 1
    assert not wired.searches


def test_list_shows_every_published_number(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list"]) == 0
    printed = capsys.readouterr().out
    for shown in ("48.5", "35.6", "94.0", "92.0", "66.2", "94.6", "55.4", "91.6", "63.49"):
        assert shown in printed
