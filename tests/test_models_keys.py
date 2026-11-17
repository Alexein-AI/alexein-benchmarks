from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from alexein_bench import keys
from alexein_bench.models import Bill, ModelError, Models, provider_of

KEY_TYPED = "tm_" + "t" * 43
KEY_FILE = "tm_" + "f" * 43


def test_provider_follows_the_key() -> None:
    assert provider_of("sk-or-v1-abc") == "openrouter"
    assert provider_of("sk-proj-abc") == "openai"


def test_an_openai_key_refuses_other_models(tmp_path: Path) -> None:
    models = Models("sk-abc", tmp_path)
    models.check(("openai/gpt-4.1-mini",))
    with pytest.raises(ModelError, match="OpenRouter"):
        models.check(("openai/gpt-4.1-mini", "google/gemini-2.5-flash"))


def _models(tmp_path: Path, key: str, sent: list[dict[str, object]]) -> Models:
    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hi"}}], "usage": usage}
        )

    models = Models(key, tmp_path)
    models._http = httpx.Client(transport=httpx.MockTransport(handle))
    return models


def test_replies_are_kept_and_reused(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []
    models = _models(tmp_path, "sk-abc", sent)
    msg = [{"role": "user", "content": "q"}]
    assert models.chat("openai/gpt-4.1-mini", msg) == "hi"
    assert models.chat("openai/gpt-4.1-mini", msg) == "hi"
    assert len(sent) == 1
    assert sent[0]["model"] == "gpt-4.1-mini"
    assert models.bill.cached == 1
    assert models.bill.usd == pytest.approx((1000 * 0.40 + 100 * 1.60) / 1e6)


def test_without_reuse_every_call_is_asked(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []
    models = _models(tmp_path, "sk-or-abc", sent)
    msg = [{"role": "user", "content": "q"}]
    models.chat("openai/gpt-4.1-mini", msg, temperature=0.7, reuse=False)
    models.chat("openai/gpt-4.1-mini", msg, temperature=0.7, reuse=False)
    assert len(sent) == 2
    assert sent[0]["model"] == "openai/gpt-4.1-mini"
    assert sent[0]["temperature"] == 0.7


def test_bill_uses_the_providers_cost_when_given() -> None:
    bill = Bill()
    bill.add("openai/gpt-4.1-mini", {"prompt_tokens": 10, "cost": 0.5}, cached=False)
    assert bill.usd == 0.5


def test_keys_come_from_the_environment_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "alexein-bench").mkdir()
    (tmp_path / "alexein-bench" / "keys.env").write_text(f"ALEXEIN_API_KEY={KEY_FILE}\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    monkeypatch.delenv("ALEXEIN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert keys.load() == (KEY_FILE, "sk-or-env")


def test_a_missing_key_off_a_terminal_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for name in ("ALEXEIN_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="Missing key"):
        keys.load()


def test_keys_are_saved_readable_by_you_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for name in ("ALEXEIN_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter([KEY_TYPED, "sk-or-typed"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))
    assert keys.load() == (KEY_TYPED, "sk-or-typed")
    path = tmp_path / "alexein-bench" / "keys.env"
    assert path.stat().st_mode & 0o777 == 0o600
    assert "OPENROUTER_API_KEY=sk-or-typed" in path.read_text()


def test_reasoning_models_take_max_completion_tokens(tmp_path: Path) -> None:
    sent: list[dict[str, object]] = []
    models = _models(tmp_path, "sk-abc", sent)
    msg = [{"role": "user", "content": "q"}]
    models.chat("openai/gpt-5", msg, temperature=None, max_tokens=4096)
    models.chat("openai/gpt-4.1-mini", msg, max_tokens=50)
    assert sent[0]["max_completion_tokens"] == 4096
    assert "max_tokens" not in sent[0]
    assert "temperature" not in sent[0]
    assert sent[1]["max_tokens"] == 50


KEY = "tm_" + "a" * 40 + "-_Z"


def test_a_key_pasted_three_times_is_one_key() -> None:
    assert keys.alexein_key(KEY * 3) == KEY
    assert keys.alexein_key(f"  {KEY}\n") == KEY
    assert keys.alexein_key("tm_short") is None
    assert keys.once("sk-or-v1-xyzsk-or-v1-xyz") == "sk-or-v1-xyz"
    assert keys.once("sk-abc") == "sk-abc"


def test_a_kept_key_pasted_twice_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for name in ("ALEXEIN_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "alexein-bench").mkdir()
    path = tmp_path / "alexein-bench" / "keys.env"
    path.write_text(f"ALEXEIN_API_KEY={KEY * 2}\nOPENAI_API_KEY=sk-x\n")
    assert keys.load() == (KEY, "sk-x")
    assert f"ALEXEIN_API_KEY={KEY}\n" in path.read_text()


def test_a_wrong_key_is_asked_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for name in ("ALEXEIN_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["not-a-key", KEY, "sk-or-x"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))
    assert keys.load() == (KEY, "sk-or-x")


def test_forget_drops_one_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "alexein-bench").mkdir()
    path = tmp_path / "alexein-bench" / "keys.env"
    path.write_text(f"ALEXEIN_API_KEY={KEY}\nOPENAI_API_KEY=sk-x\n")
    keys.forget(keys.ALEXEIN)
    assert path.read_text() == "OPENAI_API_KEY=sk-x\n"


def test_a_provider_refusing_the_callers_location_is_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("alexein_bench.models.time.sleep", lambda _s: None)
    tries = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        tries["n"] += 1
        if tries["n"] == 1:
            refused = {
                "error": {"code": 400, "metadata": {"raw": "User location is not supported"}}
            }
            return httpx.Response(400, json=refused)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    models = Models("sk-or-abc", tmp_path)
    models._http = httpx.Client(transport=httpx.MockTransport(handle))
    reply = models.chat("google/gemini-2.5-flash", [{"role": "user", "content": "q"}])
    assert reply == "ok"
    assert tries["n"] == 2
