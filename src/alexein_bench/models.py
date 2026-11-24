"""The answering and grading models, on the key you bring: OpenAI's API, or OpenRouter's.

Models are named as OpenRouter names them (`openai/gpt-4.1-mini`). With an OpenAI key the
`openai/` prefix is dropped and any other model is refused before a run starts. Every reply is
kept on disk by a digest of its request, so a run that stops and starts again pays only for
what it had not asked yet.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

OPENAI = "https://api.openai.com/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
# Dollars per million tokens, input and output: only for the estimate before a run and the bill
# after it, when the provider does not say what a call cost.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-5": (1.25, 10.00),
}
# Backing off 1s, 2s, 4s ... capped at a minute: about five minutes in all, as long as an outage
# of Alexein is waited out (`memory.patient`), so a blip does not cost a long run its questions.
_RETRIES = 10
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _ROUTED_AWAY(text: str) -> bool:  # noqa: N802 - reads as the condition it is
    """A provider that refused this caller, where another may not: Google AI Studio turns away
    some countries ("User location is not supported") on a call OpenRouter could route anywhere."""
    return "location is not supported" in text


class ModelError(RuntimeError):
    pass


class ModelKeyError(ModelError):
    """The model key itself was refused - expired, revoked, out of credit, not allowed: no
    question can be answered until it is replaced, so a run stops rather than scoring each 0."""


# Refusals of the key, not of one request.
_KEY_STATUS = {401, 402, 403}


@dataclass
class Bill:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: dict[str, Any], *, cached: bool) -> None:
        with self._lock:
            self.calls += 1
            self.cached += cached
            if cached:
                return
            prompt, completion = (
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            if isinstance(usage.get("cost"), int | float):
                self.usd += float(usage["cost"])
            else:
                rate_in, rate_out = PRICES.get(model, (0.0, 0.0))
                self.usd += (prompt * rate_in + completion * rate_out) / 1e6

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": round(self.usd, 4),
        }


def provider_of(key: str) -> str:
    return "openrouter" if key.startswith("sk-or-") else "openai"


class Models:
    def __init__(self, key: str, cache: Path, *, timeout: float = 120.0) -> None:
        self.provider = provider_of(key)
        self.base = OPENROUTER if self.provider == "openrouter" else OPENAI
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.bill = Bill()
        self._local = threading.local()
        self._http = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "alexein-bench",
                **({"X-Title": "alexein-bench"} if self.provider == "openrouter" else {}),
            },
        )

    def check(self, models: tuple[str, ...]) -> None:
        """Refuse up front a model this key cannot reach."""
        if self.provider == "openai":
            other = [m for m in models if not m.startswith("openai/")]
            if other:
                raise ModelError(
                    f"{', '.join(other)} needs an OpenRouter key (sk-or-...); an OpenAI key "
                    "reaches OpenAI's models only."
                )

    @contextmanager
    def recording(self) -> Iterator[list[dict[str, Any]]]:
        """Every call this thread makes inside it, request and reply: a question's trace."""
        calls: list[dict[str, Any]] = []
        self._local.calls = calls
        try:
            yield calls
        finally:
            self._local.calls = None

    def _record(
        self, body: dict[str, Any], text: str, usage: Any, started: float, cached: bool
    ) -> None:
        calls = getattr(self._local, "calls", None)
        if calls is not None:
            calls.append(
                {
                    "request": body,
                    "reply": text,
                    "usage": usage,
                    "cached": cached,
                    "seconds": round(time.monotonic() - started, 2),
                }
            )

    def _name(self, model: str) -> str:
        return model.removeprefix("openai/") if self.provider == "openai" else model

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        json_object: bool = False,
        seed: int | None = None,
        reuse: bool = True,
    ) -> str:
        """One completion's text. `seed` only tells cached replies apart, for a judge asked
        several times at a temperature above 0. Without `reuse` a kept reply is never read
        back: a benchmark that samples is one draw per run, as its own harness is."""
        body: dict[str, Any] = {"model": self._name(model), "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            # Reasoning models take their limit as `max_completion_tokens` and refuse the other.
            reasoning = re.match(r"(openai/)?(gpt-5|o\d)", model)
            body["max_completion_tokens" if reasoning else "max_tokens"] = max_tokens
        if json_object:
            body["response_format"] = {"type": "json_object"}
        digest = hashlib.sha256(
            json.dumps({"m": model, "b": body, "s": seed}, sort_keys=True).encode()
        ).hexdigest()
        path = self.cache / digest[:2] / f"{digest}.json"
        started = time.monotonic()
        if reuse and path.exists():
            kept = json.loads(path.read_text())
            self.bill.add(model, kept.get("usage", {}), cached=True)
            self._record(body, str(kept["text"]), kept.get("usage", {}), started, cached=True)
            return str(kept["text"])
        if self.provider == "openrouter":
            body["usage"] = {"include": True}
        reply = self._post(body)
        text = str(reply["choices"][0]["message"].get("content") or "")
        usage = reply.get("usage") or {}
        self.bill.add(model, usage, cached=False)
        self._record(body, text, usage, started, cached=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "usage": usage}))
        return text

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                response = self._http.post(f"{self.base}/chat/completions", json=body)
            except httpx.TransportError as error:
                last = error
            else:
                if response.status_code == httpx.codes.OK:
                    reply: dict[str, Any] = response.json()
                    if reply.get("choices"):
                        return reply
                    last = ModelError(f"no choices in reply: {response.text[:300]}")
                elif response.status_code in _RETRY_STATUS or _ROUTED_AWAY(response.text):
                    last = ModelError(f"{response.status_code}: {response.text[:300]}")
                elif response.status_code in _KEY_STATUS:
                    raise ModelKeyError(f"{response.status_code}: {response.text[:500]}")
                else:
                    raise ModelError(f"{response.status_code}: {response.text[:500]}")
            time.sleep(min(60.0, 2**attempt + random.random()))
        raise ModelError(f"{body['model']}: {_RETRIES} attempts failed") from last


def estimate(models: tuple[str, ...], questions: int, tokens_per_question: int) -> float:
    """Dollars, roughly: every model's price averaged, a fifth of the tokens written."""
    rates = [PRICES.get(m, (1.0, 4.0)) for m in models]
    rate_in = sum(r[0] for r in rates) / len(rates)
    rate_out = sum(r[1] for r in rates) / len(rates)
    tokens = questions * tokens_per_question
    return (tokens * 0.8 * rate_in + tokens * 0.2 * rate_out) / 1e6
