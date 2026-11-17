"""An Alexein API and a model API that run in memory, behind the real clients."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from alexein import Memory

from alexein_bench.models import Bill


class FakeAlexein:
    """Holds each app user's items; a search returns every item it holds, oldest first, so a
    test can see exactly what a question was allowed to see. `slow` items are reported queued
    for that many status calls first."""

    def __init__(self, slow: int = 0, *, settles: bool = False) -> None:
        self.items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.deleted: list[str] = []
        self.searches: list[tuple[str, str, list[str]]] = []
        self.slow = slow
        # Whether it has `/v2/memory/settle`; once told, a user's items are processed at once.
        self.settles = settles
        self.settled: list[tuple[str, list[str]]] = []
        self._asked: dict[str, int] = defaultdict(int)

    def handle(self, request: httpx.Request) -> httpx.Response:  # noqa: C901, PLR0911 - one per route
        user = request.headers.get("X-Alexein-User", "")
        body = json.loads(request.content or b"{}")
        path = request.url.path
        if path == "/v2/me":
            return httpx.Response(200, json={"kind": "key", "writes": True})
        if path == "/v2/memory/add":
            self.items[user] += body["items"]
            ids = [i["id"] for i in body["items"]]
            return httpx.Response(
                202, json={"items": [{"id": i, "status": "queued"} for i in ids], "waiting": 0}
            )
        if path == "/v2/memory/status":
            out = []
            if user not in self.items:
                return httpx.Response(404, json={"detail": f"no memory for app user '{user}' yet"})
            held = {x["id"] for x in self.items[user]}
            for i in body["items"]:
                if i not in held:
                    out.append({"id": i, "status": "unknown", "reason": None})
                    continue
                self._asked[i] += 1
                done = self._asked[i] > self.slow
                # Searchable at the first look after its add; processed `slow` looks on.
                said = "processed" if done else "stored" if self._asked[i] > 1 else "queued"
                out.append({"id": i, "status": said, "reason": None})
            return httpx.Response(200, json={"items": out})
        if path == "/v2/memory/search":
            held = sorted(self.items[user], key=lambda i: i["at"])
            self.searches.append((user, body["query"], [i["id"] for i in held]))
            memories = [
                {"type": "excerpt", "ref": f"r{n}", "text": i["text"], "date": i["at"]}
                for n, i in enumerate(held)
            ]
            return httpx.Response(
                200,
                json={
                    "query": body["query"],
                    "answerable": None,
                    "notes": [],
                    "memories": memories,
                },
            )
        if path == "/v2/memory/settle" and self.settles:
            if user not in self.items:
                return httpx.Response(404, json={"detail": f"no memory for app user '{user}' yet"})
            states = {i["id"]: self._asked[i["id"]] for i in self.items[user]}
            self.settled.append((user, [i for i, n in states.items() if n > 1]))
            for i in states:
                self._asked[i] = max(self._asked[i], self.slow)
            return httpx.Response(202, json={"status": "queued", "as_of": "now"})
        if path == "/v2/memory/users/delete":
            self.deleted.append(user)
            self.items.pop(user, None)
            return httpx.Response(200, json={"items": 0, "facts": 0, "versions": 0, "waiting": 0})
        if path == "/v2/memory/users":
            users = [{"user": u, "items": len(v), "facts": 0} for u, v in self.items.items()]
            return httpx.Response(200, json={"users": users, "next": None})
        return httpx.Response(404, json={"detail": f"no route {path}"})

    def client(self) -> Memory:
        http = httpx.Client(transport=httpx.MockTransport(self.handle))
        return Memory("tm_test", base_url="https://api.test", http_client=http)


class FakeModels:
    """Replies with `reply(model, prompt)`; counts calls."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.recorded: list[dict[str, Any]] | None = None
        self.bill = Bill()

    @contextmanager
    def recording(self) -> Iterator[list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []
        self.recorded = calls
        try:
            yield calls
        finally:
            self.recorded = None

    def chat(self, model: str, messages: list[dict[str, str]], **_: Any) -> str:
        prompt = messages[-1]["content"]
        self.calls.append((model, prompt))
        text = str(self.reply(model, prompt))
        if self.recorded is not None:
            self.recorded.append({"request": {"model": model, "messages": messages}, "reply": text})
        return text
