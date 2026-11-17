"""The two keys a run needs, asked for once: an Alexein key that writes, and a model key -
OpenAI's or OpenRouter's - for the model that answers and the one that grades.

Read from the environment first, then from `~/.config/alexein-bench/keys.env`; anything still
missing is asked for at the terminal and kept there, readable by you alone.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
from pathlib import Path

ALEXEIN = "ALEXEIN_API_KEY"
OPENAI = "OPENAI_API_KEY"
OPENROUTER = "OPENROUTER_API_KEY"


# An Alexein API key: tm_ and 43 more.
_ALEXEIN_KEY = re.compile(r"tm_[A-Za-z0-9_-]{43}")
_TRIES = 3


def once(value: str) -> str:
    """A key pasted more than once at a hidden prompt, as one: `abcabc` is `abc`."""
    value = value.strip()
    for size in range(1, len(value) // 2 + 1):
        if len(value) % size == 0 and value[:size] * (len(value) // size) == value:
            return value[:size]
    return value


def alexein_key(value: str) -> str | None:
    """The key, cleaned, or None if it is not one."""
    value = once(value)
    return value if _ALEXEIN_KEY.fullmatch(value) else None


def forget(name: str) -> None:
    """Drop a kept key, so the next run asks for it again."""
    path = config_dir() / "keys.env"
    kept = _read(path)
    if kept.pop(name, None) is not None:
        _save(path, kept)


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "alexein-bench"


def _read(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and not name.startswith("#"):
            out[name.strip()] = value.strip().strip("'\"")
    return out


def _save(path: Path, keys: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text("".join(f"{k}={v}\n" for k, v in sorted(keys.items())))


def _ask(prompt: str) -> str:
    if not sys.stdin.isatty():
        raise SystemExit(f"Missing key: set it in the environment ({prompt}).")
    value = getpass.getpass(f"{prompt}: ").strip()
    if not value:
        raise SystemExit("No key given.")
    return value


def _ask_alexein() -> str:
    for _ in range(_TRIES):
        key = alexein_key(_ask(ALEXEIN))
        if key:
            return key
        print("That is not an Alexein key: one is tm_ and 43 letters, digits, - or _.")
    raise SystemExit("No Alexein key given.")


def load() -> tuple[str, str]:
    """(Alexein key, model key)."""
    path = config_dir() / "keys.env"
    kept = _read(path)
    found = {**kept, **{k: v for k in (ALEXEIN, OPENAI, OPENROUTER) if (v := os.environ.get(k))}}
    changed = False
    given = found.get(ALEXEIN)
    if given is not None:
        cleaned = alexein_key(given)
        if cleaned is None and os.environ.get(ALEXEIN):
            raise SystemExit(f"{ALEXEIN} in your environment is not an Alexein key (tm_...).")
        changed = cleaned != given
        found[ALEXEIN] = cleaned or ""
    if not found.get(ALEXEIN):
        print("An Alexein key that reads and writes (tm_...), from app.alexein.ai/console.")
        print("Give the benchmark a project of its own: it makes and erases app users there.")
        found[ALEXEIN] = _ask_alexein()
        changed = True
    for name in (OPENROUTER, OPENAI):
        if found.get(name) and once(found[name]) != found[name]:
            found[name] = once(found[name])
            changed = True
    if not (found.get(OPENROUTER) or found.get(OPENAI)):
        print("A model key for the answering and grading models: OpenAI (sk-...) or OpenRouter")
        print("(sk-or-...). Some benchmarks' models are reached through OpenRouter only.")
        key = once(_ask("Model key"))
        found[OPENROUTER if key.startswith("sk-or-") else OPENAI] = key
        changed = True
    if changed:
        _save(path, {k: v for k, v in found.items() if k in (ALEXEIN, OPENAI, OPENROUTER)})
        print(f"Kept in {path}.")
    return found[ALEXEIN], found.get(OPENROUTER) or found[OPENAI]
