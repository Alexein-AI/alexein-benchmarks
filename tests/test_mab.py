"""MemoryAgentBench: units from fixture rows, and the scorer, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alexein_bench.benches import mab
from alexein_bench.core import Question


def _row(source: str, context: str, questions: list[str], answers: list[list[str]]) -> dict:
    return {
        "context": context,
        "questions": questions,
        "answers": answers,
        "metadata": {"source": source, "qa_pair_ids": [], "question_types": []},
    }


def test_facts_are_one_item_each_in_serial_order() -> None:
    context = "Here is a list of facts:\n0. Paris is in France.\n1. Paris is in Spain.\nnoise"
    row = _row("factconsolidation_sh_262k", context, ["Where is Paris?"], [["Spain"]])
    unit = mab.unit_of(row, 0, mab.SUBSETS["fc"])
    assert [i["text"] for i in unit.items] == ["0. Paris is in France.", "1. Paris is in Spain."]
    assert unit.items[0]["at"] < unit.items[1]["at"]
    assert all(i["kind"] == "message" and i["role"] == "user" for i in unit.items)
    q = unit.questions[0]
    assert q.cutoff is None
    assert q.extra["answers"] == ["Spain"]
    assert q.extra["family"] == "factconsolidation"


def test_windows_pack_whole_sentences() -> None:
    text = " ".join(f"Sentence number {n} is here." for n in range(400))
    chunks = mab.windows(text, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 * mab.CHARS_PER_TOKEN + 40 for c in chunks)
    assert " ".join(chunks) == text


def test_search_query_drops_the_preamble() -> None:
    assert mab.query_of("Story...\n Now Answer the Question: Who did it?") == "Who did it?"
    assert mab.query_of("Who did it?") == "Who did it?"


def test_substring_and_exact_match() -> None:
    got = mab.post_process(
        "Answer: The Federal University of Rio", ["Federal University of Rio"], "ruler_qa"
    )
    assert got["substring_exact_match"] == 1.0
    assert got["exact_match"] == 1.0
    assert mab.post_process("Lyon", ["Paris"], "ruler_qa")["substring_exact_match"] == 0.0


def test_detective_reads_the_json_answer() -> None:
    output = '{"answer": "C. The Brandt couple", "reasoning": "..."}'
    got = mab.post_process(output, ["C. The Brandt couple"], "detective_qa")
    assert got["exact_match"] == 1.0


class _Models:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kw: object) -> str:
        self.calls.append({"model": model, "messages": messages, **kw})
        return self.reply


def test_ask_uses_the_benchmarks_template_and_settings() -> None:
    q = Question(
        id="ruler_qa1_197K-0/1",
        question="In what country is Normandy located?",
        answer="",
        category="ruler_qa",
        extra={"answers": ["France"], "family": "ruler_qa", "metric": "substring_exact_match"},
    )
    models = _Models("France")
    row = mab.Mab().ask(models, q, "CONTEXT")  # type: ignore[arg-type]
    assert row["score"] == 1.0
    call = models.calls[0]
    assert call["model"] == "openai/gpt-4.1-mini"
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 50
    assert call["messages"][1]["content"].startswith("CONTEXT\nAnswer the question based on")


def test_summary_leaves_failed_rows_out() -> None:
    rows = [{"score": 1.0}, {"score": 0.0}, {"error": "boom"}]
    assert mab.Mab().summary(rows, "fc") == {"fc": "50.0"}


def test_units_read_the_converted_split(tmp_path: Path) -> None:
    rows = [
        _row("ruler_qa1_197K", "One. Two.", ["Q?"], [["One"]]),
        _row("ruler_qa2_421K", "Other.", ["Q?"], [["x"]]),
    ]
    (tmp_path / "Accurate_Retrieval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    units = list(mab.Mab().units(tmp_path, "ruler"))
    assert [u.id for u in units] == ["ruler_qa1_197K-0"]
    assert units[0].items[0]["text"] == "One. Two."


@pytest.mark.parametrize("variant", mab.Mab.variants)
def test_every_variant_has_a_subset(variant: str) -> None:
    assert variant in mab.SUBSETS
