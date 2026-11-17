"""The README's charts, drawn from `results/*.json` (Alexein's numbers) and `assets/others.json`
(everyone else's, as their sources publish them), so a chart never shows a number the results
don't hold. Each title's claim is checked against the numbers it draws: a result that stops
supporting one fails here, not in front of a reader.

python scripts/readme_charts.py          redraw assets/<chart>-light.svg and -dark.svg
python scripts/readme_charts.py --check  exit 1 if one is out of date

Drawn as alexein.ai/benchmarks draws them: Alexein in ink, memory systems a quiet grey, and a
model with no memory system (reading the whole history, or plain retrieval) as a pale bar with a
fine edge; every bar from 0 in a track to 100, so a small gap looks as small as it is.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
sys.path.insert(0, str(ROOT / "src"))

from alexein_bench.published import Row, boards  # noqa: E402

OTHERS: dict[str, Any] = json.loads((ASSETS / "others.json").read_text())
RESULTS: dict[tuple[str, str], Row] = {(b.bench, r.key): r for b in boards() for r in b.rows}
TITLES = {b.bench: b.title for b in boards()}


# --- Colour. GitHub's two page colours, and the site's chart tones mixed against each (in OKLab,
# as alexein.ai mixes them): other 24% ink, ghost 13% with a 30% edge, track 5%, notch 11%.


@dataclass(frozen=True)
class Theme:
    name: str
    page: str
    edge: str  # the card's hairline
    ink: str  # Alexein's bars, titles and figures
    text: str
    muted: str
    faint: str
    other: str
    ghost: str
    ghost_edge: str
    track: str
    notch: str
    row: str  # the band behind Alexein's row
    tile: str


LIGHT = Theme(
    "light", page="#ffffff", edge="#e5e5e3", ink="#0a0a0a", text="#1d1d20", muted="#6e6e73",
    faint="#86868b", other="#bcbcbc", ghost="#dadada", ghost_edge="#acacac", track="#f1f1f1",
    notch="#e0e0e0", row="#f5f5f4", tile="#f6f6f5",
)  # fmt: skip
DARK = Theme(
    "dark", page="#0d1117", edge="#2a2f37", ink="#f5f5f5", text="#e4e4e6", muted="#9a9aa0",
    faint="#6e6e74", other="#4a4e55", ghost="#25292f", ghost_edge="#4f5359", track="#161a20",
    notch="#24282e", row="#151a21", tile="#151a21",
)  # fmt: skip
# The site's four pastels, one to a headline figure.
PASTELS = ("#c3ceff", "#bfe7d2", "#ffd2bc", "#dccaf5")

FONT = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', "
    "Helvetica, Arial, sans-serif"
)
W = 880  # every chart's width: about the README's column on github.com
P = 32  # a card's inner margin
TIMES = chr(0xD7)  # the multiplication sign, which a linter reads as a stray "x"


def holds(ok: bool, what: str) -> None:
    if not ok:
        raise SystemExit(f"A chart's claim no longer holds: {what}")


def score(shown: str) -> float:
    return float(shown.rstrip("%"))


def model(name: str) -> str:
    """A results file's model as the charts write it: "gpt-4.1-mini" is "GPT-4.1-mini"."""
    return "GPT" + name[3:] if name.startswith("gpt-") else name


def wide(text: str, size: float) -> float:
    """A generous guess at a line's width, so text set from it never collides in a wider face."""
    return len(text) * size * 0.56


def wrap(text: str, width: float, size: float) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and wide(f"{lines[-1]} {word}", size) <= width:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines


# --- Drawing.


def text(x: float, y: float, words: str, cls: str, anchor: str = "start") -> str:
    end = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:g}" y="{y:g}" class="{cls}"{end}>{escape(words)}</text>'


def bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Square at 0, rounded at its end."""
    r = min(r, w / 2, h / 2)
    return (
        f"M{x:.1f} {y:.1f}H{x + w - r:.1f}A{r:g} {r:g} 0 0 1 {x + w:.1f} {y + r:.1f}"
        f"V{y + h - r:.1f}A{r:g} {r:g} 0 0 1 {x + w - r:.1f} {y + h:.1f}H{x:.1f}Z"
    )


def bar(t: Theme, x: float, y: float, w: float, h: float, *, kind: str) -> str:
    fill = {"ours": t.ink, "memory": t.other, "none": t.ghost}[kind]
    out = f'<path d="{bar_path(x, y, w, h)}" fill="{fill}"/>'
    if kind == "none" and w > 1:
        inner = bar_path(x + 0.5, y + 0.5, w - 1, h - 1, 3.5)
        out += f'<path d="{inner}" fill="none" stroke="{t.ghost_edge}"/>'
    return out


def track(t: Theme, x: float, y: float, w: float, h: float) -> str:
    out = [f'<path d="{bar_path(x, y, w, h)}" fill="{t.track}"/>']
    for tick in (25, 50, 75):
        nx = x + w * tick / 100
        out.append(f'<rect x="{nx - 0.5:g}" y="{y:g}" width="1" height="{h:g}" fill="{t.notch}"/>')
    return "".join(out)


def scale(x: float, y: float, w: float) -> str:
    return "".join(
        text(x + w * tick / 100, y, str(tick), "scale", "middle") for tick in (0, 25, 50, 75, 100)
    )


def legend(t: Theme, x: float, y: float, items: list[tuple[str, str]]) -> str:
    out = []
    for kind, label in items:
        out.append(bar(t, x, y - 9, 16, 10, kind=kind))
        out.append(text(x + 23, y, label, "legend"))
        x += 23 + wide(label, 12.5) + 22
    return "".join(out)


def style(t: Theme) -> str:
    return (
        f"<style>text{{font-family:{FONT};fill:{t.text}}}"
        f".eyebrow{{font-size:12.5px;fill:{t.muted};font-weight:500}}"
        f".title{{font-size:21px;font-weight:600;fill:{t.ink};letter-spacing:-0.01em}}"
        f".lead{{font-size:13.5px;fill:{t.muted}}}"
        f".legend{{font-size:12.5px;fill:{t.muted}}}"
        f".panel{{font-size:12.5px;font-weight:600;fill:{t.text}}}"
        f".name{{font-size:13px}}"
        f".aside{{font-size:12px;fill:{t.muted}}}"
        f".value{{font-size:13px;fill:{t.muted};font-variant-numeric:tabular-nums;"
        f"paint-order:stroke;stroke:{t.page};stroke-width:4px;stroke-linejoin:round}}"
        f".ours{{font-weight:600;fill:{t.ink}}}"
        f".row .value{{stroke:{t.row}}}"
        f".scale{{font-size:11px;fill:{t.faint};font-variant-numeric:tabular-nums}}"
        f".note{{font-size:12px;fill:{t.muted}}}"
        f".gain{{font-size:17px;font-weight:600;fill:{t.ink}}}"
        f".figure{{font-size:40px;font-weight:600;fill:{t.ink};letter-spacing:-0.02em}}"
        f".label{{font-size:14px;font-weight:500;fill:{t.ink}}}"
        "</style>"
    )


def svg(
    t: Theme, height: float, body: list[str], title: str, desc: str, *, card: bool = True
) -> str:
    frame = (
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1:g}" rx="16" '
        f'fill="{t.page}" stroke="{t.edge}"/>'
        if card
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height:g}" '
        f'viewBox="0 0 {W} {height:g}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>'
        f"{style(t)}{frame}{''.join(body)}</svg>\n"
    )


@dataclass(frozen=True)
class Header:
    eyebrow: str
    title: str
    lead: str
    legend: list[tuple[str, str]]


def header(t: Theme, h: Header) -> tuple[list[str], float]:
    # Checked in a wide face (Verdana's letters run about 0.62em), not the one it's set in.
    holds(len(h.title) * 21 * 0.62 <= W - 2 * P, f"the title fits: {h.title}")
    out = [text(P, P + 14, h.eyebrow, "eyebrow"), text(P, P + 44, h.title, "title")]
    y = P + 44
    for line in wrap(h.lead, W - 2 * P, 13.5):
        y += 21
        out.append(text(P, y, line, "lead"))
    if h.legend:
        y += 32
        out.append(legend(t, P, y, h.legend))
    return out, y + 26


def notes(lines: list[str], y: float) -> tuple[list[str], float]:
    out = []
    for note in lines:
        for line in wrap(note, W - 2 * P, 12):
            y += 18
            out.append(text(P, y, line, "note"))
        y += 4
    return out, y


# --- Boards: one bar per system, highest first, Alexein's row among them.


@dataclass(frozen=True)
class Bar:
    name: str
    shown: str
    kind: str  # ours, memory or none
    aside: str = ""


@dataclass(frozen=True)
class Panel:
    heading: str
    rows: list[Bar]


LABELS = 300  # the names' column
VALUE = 64  # room for a value past the end of the scale
BX = P + LABELS
BW = W - P - VALUE - BX
ROW = 30


def board(ref: str, aside: str = "") -> list[Bar]:
    """`results/<bench>.json`'s row beside the systems `assets/others.json` lists for it."""
    bench, key = ref.split("/")
    rows = [
        Bar(r["name"], r["shown"], r["kind"], r.get("aside", "")) for r in OTHERS["boards"][ref]
    ]
    rows.append(Bar("Alexein", RESULTS[bench, key].shown, "ours", aside))
    return sorted(rows, key=lambda r: -score(r.shown))


def above(rows: list[Bar]) -> list[Bar]:
    """The rows ahead of Alexein's."""
    return rows[: next(i for i, r in enumerate(rows) if r.kind == "ours")]


def row(t: Theme, r: Bar, y: float) -> list[str]:
    ours = r.kind == "ours"
    holds(wide(r.name, 13) + wide(r.aside, 12) + 20 <= LABELS, f"{r.name} fits its column")
    out = ['<g class="row">' if ours else "<g>"]
    if ours:
        out.append(
            f'<rect x="{P - 12}" y="{y - ROW / 2 + 1:g}" width="{W - 2 * P + 24}" '
            f'height="{ROW - 2}" rx="8" fill="{t.row}"/>'
        )
    out.append(text(P, y + 4.5, r.name, "name ours" if ours else "name"))
    if r.aside:
        out.append(text(BX - 16, y + 4.5, r.aside, "aside", "end"))
    out.append(track(t, BX, y - 6, BW, 12))
    w = BW * score(r.shown) / 100
    out.append(bar(t, BX, y - 6, w, 12, kind=r.kind))
    out.append(text(BX + w + 9, y + 4.5, r.shown, "value ours" if ours else "value"))
    return [*out, "</g>"]


def boards_chart(t: Theme, h: Header, panels: list[Panel], foot: list[str], desc: str) -> str:
    body, y = header(t, h)
    for n, panel in enumerate(panels):
        if panel.heading:
            y += 14 if n else 0
            body.append(text(P, y + 4, panel.heading, "panel"))
            y += 20
        for r in panel.rows:
            body += row(t, r, y + ROW / 2)
            y += ROW
    body.append(scale(BX, y + 18, BW))
    lines, y = notes(foot, y + 30)
    return svg(t, y + P - 6, body + lines, f"{h.eyebrow}: {h.title}", desc)


def described(panels: list[Panel]) -> str:
    return " ".join(
        (f"{p.heading}: " if p.heading else "")
        + "; ".join(f"{r.name} {r.shown}" for r in p.rows)
        + "."
        for p in panels
    )


MEMORY = ("memory", "Memory systems")
NONE = ("none", "No memory system: the whole history, or plain retrieval")
OURS = ("ours", "Alexein")


def rhelm(t: Theme) -> str:
    with_ = board("rhelm/with")
    without = board("rhelm/without")
    for rows in (with_, without):
        holds(all(r.kind == "none" for r in above(rows)), "RHELM: no memory framework is ahead")
    rag = next(r for r in above(without) if "retrieval" in r.name)
    ours = next(r for r in without if r.kind == "ours")
    holds(score(rag.shown) - score(ours.shown) < 2.6, "RHELM: the retrieval row is in the margin")
    frameworks = list(dict.fromkeys(r.name for r in with_ + without if r.kind == "memory"))
    panels = [Panel("Chats, emails and documents", with_), Panel("Chats only", without)]
    h = Header(
        f"RHELM · {RESULTS['rhelm', 'with'].questions} questions",
        "The top memory framework on both tables",
        "Microsoft's benchmark of a year in ten people's lives. "
        f"{', '.join(frameworks[:-1])} and {frameworks[-1]} read with GPT-4.1-mini, as Alexein "
        "does. Ahead of it: frontier models reading the whole history in a 1M-token window, and "
        "on chats alone one retrieval setup, within the margin.",
        [OURS, MEMORY, NONE],
    )
    foot = [
        "Averages as published on the RHELM leaderboard; Alexein's are self-reported. "
        "The margin on an average is ±2.6 at 95%."
    ]
    return boards_chart(t, h, panels, foot, described(panels))


def mab(t: Theme) -> str:
    mine = RESULTS["mab", "fc"]
    rows = board("mab/fc", model(mine.reader))
    holds(rows[0].kind == "ours", "MemoryAgentBench: first on fact consolidation")
    panels = [Panel("", rows)]
    h = Header(
        f"MemoryAgentBench · {mine.label}",
        "First when the facts keep changing",
        "262K tokens of facts, each overwritten as the history goes on; the right answer is the "
        "latest. Every row is scored by exact match, with no judge, and answers with the model "
        "beside it, or the one its name says.",
        [OURS, MEMORY, NONE],
    )
    foot = [
        f"Alexein's is self-reported, on {mine.questions} questions. The rest are as the "
        "MemoryAgentBench paper and MindMemOS publish them."
    ]
    return boards_chart(t, h, panels, foot, described(panels))


def locomo(t: Theme) -> str:
    mine = RESULTS["locomo", "all"]
    rows = board("locomo/all")
    holds(rows[0].kind == "ours", "LoCoMo: at the top")
    margin = score(rows[0].shown) - score(rows[1].shown)
    holds(margin < 1, "LoCoMo: by under a point")
    panels = [Panel("", rows)]
    h = Header(
        f"LoCoMo · {mine.questions} questions",
        "At the top, by under a point",
        "Long conversations over many sessions, then questions about what was said. Every row "
        f"answers with {model(mine.reader)}, so the memory is what's compared.",
        [OURS, MEMORY, NONE],
    )
    foot = [
        f"{margin:.2f} points separate Alexein and {rows[1].name}. Judges vary by source and the "
        "answer key has known errors, so read the top of this board as close. Alexein's is "
        "self-reported; the rest are as the Mnemis and MindMemOS papers publish them."
    ]
    return boards_chart(t, h, panels, foot, described(panels))


def plus_ratio() -> tuple[Bar, str]:
    rows = board("locomo_plus/all")
    best = next(r for r in rows if r.kind == "memory")
    ours = next(r for r in rows if r.kind == "ours")
    holds(rows[0] is ours, "LoCoMo-Plus: ahead of the paper's every row")
    return best, f"{score(ours.shown) / score(best.shown):.1f}{TIMES}"


def locomo_plus(t: Theme) -> str:
    mine = RESULTS["locomo_plus", "all"]
    rows = board("locomo_plus/all", model(mine.reader))
    best, ratio = plus_ratio()
    panels = [Panel("", rows)]
    h = Header(
        f"LoCoMo-Plus · {mine.questions} questions",
        f"{ratio} the paper's best memory system",
        "Cues placed in LoCoMo conversations, asked about long after, with a binary judge. "
        "The paper's Table 1, with Alexein's run added: its memory systems answer with GPT-4o, "
        f"Alexein with {model(mine.reader)}.",
        [OURS, MEMORY, NONE],
    )
    foot = [
        f"{mine.shown} against {best.name}'s {best.shown}. Alexein's is self-reported; the rest "
        "are the LoCoMo-Plus paper's Table 1."
    ]
    return boards_chart(t, h, panels, foot, described(panels))


def longmemeval(t: Theme) -> str:
    mine = RESULTS["longmemeval", "all"]
    rows = board("longmemeval/all")
    ahead = above(rows)
    holds(len(ahead) == 1 and rows[-1].kind != "ours", "LongMemEval-S: second of three")
    gap = score(ahead[0].shown) - score(mine.shown)
    panels = [Panel("", rows)]
    h = Header(
        f"LongMemEval-S · {mine.questions} questions",
        "Close, and not first",
        f"Ahead of Mem0's open-source release, run in the same setup; {gap:.1f} points behind "
        "what Mem0's hosted platform reports on its blog. Other vendors self-report higher still.",
        [],
    )
    foot = [f"Alexein's is self-reported, answering with {mine.reader} and graded by {mine.judge}."]
    return boards_chart(t, h, panels, foot, described(panels))


def beam(t: Theme) -> str:
    mean, rate = board("beam/mean"), board("beam/pass")
    gaps = [abs(score(a.shown) - score(b.shown)) for a, b in (mean, rate)]
    holds(all(g < 1 for g in gaps), "BEAM: level with Mem0")
    panels = [Panel("Mean score", mean), Panel("Pass rate", rate)]
    mine = RESULTS["beam", "mean"]
    h = Header(
        f"BEAM · 1M-token split · {mine.questions} questions",
        "A million tokens a conversation, level with Mem0",
        "35 conversations of about a million tokens each, every one larger than GPT-5's "
        "400,000-token window. GPT-5 answers and judges, in Mem0's published harness, no hints.",
        [],
    )
    foot = [
        f"Within a point on both: {gaps[0]:.2f} on the mean, {gaps[1]:.2f} on the pass rate. "
        "Alexein's are self-reported; Mem0's are from its published harness."
    ]
    return boards_chart(t, h, panels, foot, described(panels))


# --- Same model, better memory: GPT-4.1-mini reading the whole history, then reading Alexein.

SAME = "GPT-4.1-mini"


@dataclass(frozen=True)
class Pair:
    name: str
    task: str
    whole: str
    ours: str
    judged_apart: bool

    @property
    def gain(self) -> float:
        return score(self.ours) - score(self.whole)


def pairs() -> list[Pair]:
    out = []
    for p in OTHERS["same_model"]:
        mine = RESULTS[p["bench"], p["key"]]
        holds(model(mine.reader) == SAME, f"Same model: {p['name']} answers with {SAME}")
        out.append(Pair(p["name"], p["task"], p["whole"], mine.shown, p.get("judged_apart", False)))
    holds(all(p.gain > 0 for p in out), "Same model: better with Alexein in every pair")
    return sorted(out, key=lambda p: -p.gain)


def same_model(t: Theme) -> str:
    labels, gain_w, value = 236, 64, 56
    bx = P + labels
    bw = W - P - gain_w - value - bx
    h = Header(
        "Same model, better memory",
        "The same model scores higher with Alexein",
        f"Each pair is the same questions, the same scoring and the same answer model, {SAME}: "
        "once with the whole history in its context, once with what Alexein returns for the "
        "question.",
        [("none", f"{SAME}, the whole history"), ("ours", f"{SAME} with Alexein")],
    )
    body, y = header(t, h)
    body.append(text(W - P, y - 2, "points", "scale", "end"))
    y += 10
    for n, p in enumerate(pairs()):
        if n:
            body.append(
                f'<rect x="{P}" y="{y - 10:g}" width="{W - 2 * P}" height="1" fill="{t.edge}"/>'
            )
        star = "*" if p.judged_apart else ""
        body.append(text(P, y + 13, p.name + star, "name ours"))
        body.append(text(P, y + 31, p.task, "aside"))
        for i, (shown, kind) in enumerate(((p.whole, "none"), (p.ours, "ours"))):
            by = y + 2 + i * 18
            w = bw * score(shown) / 100
            body.append(track(t, bx, by, bw, 12))
            body.append(bar(t, bx, by, w, 12, kind=kind))
            cls = "value ours" if kind == "ours" else "value"
            body.append(text(bx + w + 9, by + 10.5, shown, cls))
        body.append(text(W - P, y + 24, f"+{p.gain:.1f}", "gain", "end"))
        y += 58
    body.append(scale(bx, y - 2, bw))
    judged = next(p for p in pairs() if p.judged_apart)
    mine = RESULTS["locomo", "all"]
    foot = [
        f"* {judged.name}'s whole-history score is the Mnemis paper's, graded by {SAME}; "
        f"Alexein's is graded by {mine.judge}. Whole-history scores are as MemoryAgentBench, "
        "the RHELM leaderboard and the Mnemis paper publish them; Alexein's are self-reported."
    ]
    lines, y = notes(foot, y + 10)
    desc = " ".join(
        f"{p.name}, {p.task.lower()}: {p.whole} reading everything, {p.ours} with Alexein."
        for p in pairs()
    )
    return svg(t, y + P - 6, body + lines, h.title, desc)


# --- The headline figures, four tiles.


def glance(t: Theme) -> str:
    best = pairs()[0]
    plus_best, ratio = plus_ratio()
    beam_ours, beam_mem0 = (RESULTS["beam", "pass"].shown, board("beam/pass"))
    mem0 = next(r for r in beam_mem0 if r.kind != "ours")
    frameworks = list(
        dict.fromkeys(
            r.name
            for ref in ("rhelm/with", "rhelm/without")
            for r in board(ref)
            if r.kind == "memory"
        )
    )
    tiles = [
        (
            f"+{best.gain:.1f}",
            "points, same model, better memory",
            f"{SAME} on {best.name}, {best.task.lower()}: {best.whole} to {best.ours}",
        ),
        (
            "#1",
            "memory framework on Microsoft's RHELM",
            f"Ahead of {', '.join(frameworks[:-1])} and {frameworks[-1]} on both tables, "
            f"each reading with {SAME}",
        ),
        (
            ratio,
            "the LoCoMo-Plus paper's best memory",
            f"{RESULTS['locomo_plus', 'all'].shown} against {plus_best.name}'s {plus_best.shown}",
        ),
        (
            beam_ours,
            "pass rate on BEAM's million-token conversations",
            f"Level with Mem0's {mem0.shown}, in Mem0's own harness",
        ),
    ]
    gap, tw, th = 14, (W - 14) / 2, 164
    body = []
    for n, (figure, label, note) in enumerate(tiles):
        x, y = (n % 2) * (tw + gap), (n // 2) * (th + gap)
        body.append(
            f'<rect x="{x + 0.5:g}" y="{y + 0.5:g}" width="{tw - 1:g}" height="{th - 1}" rx="16" '
            f'fill="{t.tile}" stroke="{t.edge}"/>'
        )
        body.append(
            f'<rect x="{x + 26:g}" y="{y + 24:g}" width="28" height="4" rx="2" '
            f'fill="{PASTELS[n]}"/>'
        )
        body.append(text(x + 26, y + 76, figure, "figure"))
        body.append(text(x + 26, y + 102, label, "label"))
        for i, line in enumerate(wrap(note, tw - 52, 12.5)[:2]):
            body.append(text(x + 26, y + 124 + i * 18, line, "note"))
    desc = " ".join(f"{f}: {label}. {note}." for f, label, note in tiles)
    return svg(t, 2 * th + gap, body, "Alexein's benchmarks at a glance", desc, card=False)


# --- How a run works: a question sees only what was added before it.


def timeline(t: Theme) -> str:
    h = Header(
        "How a run works",
        "A question sees only what came before it",
        "Every item is added in time order, with its own date. A question is asked once "
        "everything dated before it has been added and processed, and before anything dated on "
        "or after it is added.",
        [],
    )
    body, y = header(t, h)
    left, right = P + 8, W - P - 8
    axis = y + 76
    asked = left + (right - left) * 0.52
    days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    body.append(
        f'<rect x="{left}" y="{axis - 0.5:g}" width="{right - left}" height="1" fill="{t.edge}"/>'
    )
    for i, day in enumerate(days):
        dx = left + (right - left) * (i + 0.5) / len(days)
        body.append(text(dx, axis + 30, day, "scale", "middle"))
    # Items through the week, irregular as a real history is.
    spots = (0.02, 0.06, 0.09, 0.15, 0.2, 0.23, 0.31, 0.35, 0.37, 0.44, 0.5, 0.53, 0.58, 0.6,
             0.67, 0.71, 0.74, 0.81, 0.86, 0.9, 0.96)  # fmt: skip
    for s in spots:
        cx = left + (right - left) * s
        if cx < asked:
            body.append(
                f'<circle cx="{cx:.1f}" cy="{axis:g}" r="6" fill="{t.ink}" '
                f'stroke="{t.page}" stroke-width="2"/>'
            )
        else:
            body.append(
                f'<circle cx="{cx:.1f}" cy="{axis:g}" r="5.5" fill="{t.ghost}" '
                f'stroke="{t.ghost_edge}" stroke-width="1"/>'
            )
    body.append(
        f'<rect x="{asked - 1:g}" y="{axis - 44:g}" width="2" height="58" rx="1" fill="{t.ink}"/>'
    )
    body.append(
        text(asked, axis - 52, "A question, asked Thursday afternoon", "name ours", "middle")
    )
    body.append(
        text((left + asked) / 2, axis + 52, "Added and processed before it", "aside", "middle")
    )
    body.append(
        text((asked + right) / 2, axis + 52, "Added only after it is answered", "aside", "middle")
    )
    return svg(t, axis + 52 + P, body, h.title, h.lead)


CHARTS: dict[str, Callable[[Theme], str]] = {
    "glance": glance,
    "same-model": same_model,
    "rhelm": rhelm,
    "memoryagentbench": mab,
    "locomo": locomo,
    "locomo-plus": locomo_plus,
    "longmemeval": longmemeval,
    "beam": beam,
    "how-a-run-works": timeline,
}


def drawn() -> dict[Path, str]:
    return {
        ASSETS / f"{name}-{theme.name}.svg": draw(theme)
        for name, draw in CHARTS.items()
        for theme in (LIGHT, DARK)
    }


def main() -> int:
    want = drawn()
    if "--check" in sys.argv:
        stale = [p for p, s in want.items() if not p.is_file() or p.read_text() != s]
        for path in stale:
            print(f"{path.relative_to(ROOT)} is out of date: python scripts/readme_charts.py")
        return 1 if stale else 0
    for path, content in want.items():
        path.write_text(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
