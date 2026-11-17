"""`alexein-bench`: list the published numbers, run a benchmark, clean up after one.

alexein-bench list
alexein-bench run rhelm                     # every variant; asks before spending
alexein-bench run rhelm --variant with --limit 20
alexein-bench run rhelm --resume <run-id>
alexein-bench clean [--run <run-id>]
"""

from __future__ import annotations

import argparse
import datetime as dt
import secrets
import sys
from pathlib import Path

from alexein import AlexeinError, AuthenticationError, Memory

from alexein_bench import keys, published
from alexein_bench.benches import BENCHES
from alexein_bench.core import Bench
from alexein_bench.fetch import cache_dir
from alexein_bench.memory import leftovers, patient, short
from alexein_bench.models import Models, estimate
from alexein_bench.runner import Runner, Settings, trimmed


def _say(line: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {line}", flush=True)


def cmd_list(_: argparse.Namespace) -> int:
    for board in published.boards():
        runnable = "" if board.bench in BENCHES else "  (no harness yet)"
        print(f"{board.bench:12} {board.title}{runnable}")
        for row in board.rows:
            command = f"alexein-bench run {board.bench} --variant {row.variant}"
            print(f"    {row.label:42} {row.shown:>8}   {command}")
    return 0


def _confirm(bench: Bench, units: int, questions: int, items: int, assume_yes: bool) -> bool:
    usd = estimate(bench.models, questions, bench.tokens_per_question)
    print(
        f"\n{bench.title}: {units} app users, {items:,} items added, {questions:,} searches.\n"
        f"Alexein: {items + units:,} events and {questions:,} retrievals from your plan.\n"
        f"Models ({', '.join(bench.models)}): about ${usd:,.2f} on your key.\n"
    )
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Not a terminal: pass --yes to run.")
        return False
    return input("Run it? [y/N] ").strip().lower() in ("y", "yes")


def _report(bench: Bench, variant: str, summary: dict[str, object]) -> None:
    board = published.board(bench.name)
    got = summary.get("scores") or {}
    assert isinstance(got, dict)
    print(
        f"\n{bench.title} ({variant})  -  {summary['answered']} of {summary['questions']} answered"
    )
    print(f"  {'':42} {'published':>10} {'this run':>10}")
    for row in board.rows:
        if row.variant != variant:
            continue
        print(f"  {row.label:42} {row.shown:>10} {got.get(row.key, '-'):>10}")


def _key_works(memory: Memory) -> bool:
    """A wrong key stops here, before anything is spent."""
    try:
        me = patient(memory.me)
    except AuthenticationError:
        keys.forget(keys.ALEXEIN)
        print(
            f"{memory.base_url} refused the Alexein key (401). It is no longer kept: run again"
            f" to enter it, or check {keys.ALEXEIN} if it is set in your environment."
        )
        return False
    except AlexeinError as error:
        print(f"{memory.base_url} could not be reached: {short(error)}. Try again shortly.")
        return False
    if not me.writes:
        print("This Alexein key only reads: a run adds items, so it needs a key that writes.")
        return False
    return True


def cmd_run(args: argparse.Namespace) -> int:
    bench = BENCHES[args.bench]
    variants = [args.variant] if args.variant else list(bench.variants)
    for variant in variants:
        if variant not in bench.variants:
            print(f"{bench.name} has no variant {variant}: {', '.join(bench.variants)}")
            return 2
    alexein_key, model_key = keys.load()
    cache = cache_dir()
    models = Models(model_key, cache / "replies")
    models.check(bench.models)
    data = bench.fetch(cache / "data")
    memory = Memory(alexein_key, base_url=args.api_url)
    if not _key_works(memory):
        return 2

    planned = []
    for variant in variants:
        units = trimmed(bench.units(data, variant), args.limit, args.units)
        planned.append((variant, units))
    n_units = sum(len(u) for _, u in planned)
    n_questions = sum(len(x.questions) for _, units in planned for x in units)
    n_items = sum(len(x.items) for _, units in planned for x in units)
    if not _confirm(bench, n_units, n_questions, n_items, args.yes):
        return 1

    run = args.resume or f"{dt.date.today():%Y%m%d}{secrets.token_hex(3)}"
    for variant, units in planned:
        out = Path(args.out) / bench.name / variant / run
        out.mkdir(parents=True, exist_ok=True)
        settings = Settings(
            bench=bench, variant=variant, run=run, out=out, workers=args.workers,
            wait=args.wait, keep=args.keep, resume=bool(args.resume), report_every=args.every,
            heartbeat=args.status,
        )  # fmt: skip
        _say(f"run {run}: {bench.title} ({variant}) -> {out}")
        summary = Runner(settings, memory, models).run(units, _say)
        _report(bench, variant, summary)
    bill = models.bill
    print(f"\nModels: ${bill.usd:,.2f} ({bill.calls} calls, {bill.cached} cached)")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    alexein_key, _ = keys.load()
    memory = Memory(alexein_key, base_url=args.api_url)
    users = leftovers(memory, args.run)
    for user in users:
        memory.user(user).delete()
        print(f"erased {user}")
    print(f"{len(users)} app users erased")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="alexein-bench", description=__doc__.split("\n\n")[0])
    sub = top.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the published numbers, and which can be run")

    run = sub.add_parser("run", help="run a benchmark against Alexein")
    run.add_argument("bench", choices=sorted(BENCHES))
    run.add_argument("--variant", help="one variant only (default: every one)")
    run.add_argument("--limit", type=int, help="at most this many questions")
    run.add_argument("--units", type=int, help="at most this many units (app users)")
    run.add_argument("--resume", metavar="RUN", help="carry on a stopped run")
    run.add_argument("--yes", action="store_true", help="do not ask before spending")
    run.add_argument("--workers", type=int, default=4, help="units at once (default 4)")
    run.add_argument("--wait", type=float, default=3600.0, help="seconds to wait for processing")
    run.add_argument("--keep", action="store_true", help="keep the app users afterwards")
    run.add_argument("--every", type=int, default=10, help="score so far every N (default 10)")
    run.add_argument(
        "--status", type=float, default=30.0, help="seconds between status lines (default 30)"
    )
    run.add_argument("--out", default="runs", help="where results go (default ./runs)")

    clean = sub.add_parser("clean", help="erase app users a run left behind")
    clean.add_argument("--run", help="one run's only")

    for p in (run, clean):
        p.add_argument("--api-url", default=None, help="Alexein API (default api.alexein.ai)")
    return top


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {"list": cmd_list, "run": cmd_run, "clean": cmd_clean}
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        print("\nStopped. Carry on with --resume and the run id above.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
