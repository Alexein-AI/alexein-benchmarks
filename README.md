# Alexein benchmarks

The numbers on [alexein.ai/benchmarks](https://alexein.ai/benchmarks), and the code to run each one
yourself against Alexein's public API, with one command and your own keys.

## Results

Every number below is written exactly as the website shows it. Alexein's results are
self-reported; the other systems' results, shown on the website, are as their own sources publish
them.

<!-- results:start -->
| Benchmark | What is scored | Alexein | Questions | Answers / grades | Run it |
|---|---|---:|---:|---|---|
| [RHELM](https://github.com/microsoft/RHELM) | Chats, emails and documents | **48.5** | 1,305 | gpt-4.1-mini / gpt-4.1-mini | `alexein-bench run rhelm --variant with` |
| [RHELM](https://github.com/microsoft/RHELM) | Chats only | **35.6** | 1,305 | gpt-4.1-mini / gpt-4.1-mini | `alexein-bench run rhelm --variant without` |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | Fact consolidation (single-hop) | **94.0** | 100 | gpt-4.1-mini / no model (exact match) | `alexein-bench run mab --variant fc` |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | RULER-QA1, accurate retrieval | **92.0** | 100 | gpt-4.1-mini / no model (exact match) | `alexein-bench run mab --variant ruler` |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | DetectiveQA, all 10 books | **66.2** | 71 | gpt-4.1-mini / no model (exact match) | `alexein-bench run mab --variant detective` |
| [LoCoMo](https://github.com/snap-research/locomo) | All questions, categories 1-4 | **94.6** | 1,540 | gpt-4.1-mini / gpt-4o-mini, majority of three | `alexein-bench run locomo --variant all` |
| [LoCoMo-Plus](https://arxiv.org/abs/2602.10715) | All questions | **55.4** | 401 | gpt-4.1-mini / the paper's judge | `alexein-bench run locomo_plus --variant all` |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) | All questions | **91.6** | 500 | gemini-3.6-flash / gpt-4o-mini | `alexein-bench run longmemeval --variant all` |
| [BEAM, 1M-token split](https://arxiv.org/abs/2510.27246) | Mean score | **63.49** | 700 | gpt-5 / gpt-5 | `alexein-bench run beam --variant 1m` |
| [BEAM, 1M-token split](https://arxiv.org/abs/2510.27246) | Pass rate | **70.57%** | 700 | gpt-5 / gpt-5 | `alexein-bench run beam --variant 1m` |
<!-- results:end -->

Every row runs with the command beside it: the same dataset, questions, prompts, models and
scoring as the published setup. Expect a score close to the published one, not the same digits.
Some benchmarks sample their models, as RHELM does, so every run is one draw. And Alexein keeps
improving, so a run today reads the memory of today.

## Run one

You need [uv](https://docs.astral.sh/uv/) and two keys:

- **An Alexein key that reads and writes** (`tm_...`) from
  [app.alexein.ai/console](https://app.alexein.ai/console). Give the benchmark a project of its
  own: a run creates app users there and erases them when it finishes.
- **A model key** for the models that answer and grade: OpenAI (`sk-...`) or OpenRouter
  (`sk-or-...`). Benchmarks whose published setup uses a model other than OpenAI's need an
  OpenRouter key, and tell you so before they start.

```sh
uvx --from git+https://github.com/Alexein-AI/alexein-benchmarks alexein-bench run rhelm
```

On the first run you are asked for both keys. They are kept in
`~/.config/alexein-bench/keys.env`, readable by you alone, or taken from `ALEXEIN_API_KEY` and
`OPENAI_API_KEY` / `OPENROUTER_API_KEY` if those are set. Before anything is spent, the run shows
how many app users, items and searches it will use from your Alexein plan and roughly what the
models will cost on your key, then asks you to confirm.

```sh
alexein-bench list                                  # every number, and the command that runs it
alexein-bench run rhelm --variant with --limit 20   # a quick check on 20 questions
alexein-bench run rhelm --resume 20261117a1b2c3     # carry on a run that stopped
alexein-bench clean                                 # erase app users a stopped run left
```

From a clone: `uv run alexein-bench ...`.

Everything a run produces goes in `runs/<benchmark>/<variant>/<run>/`, and each file is written
as the run goes, so a stopped run loses nothing it had finished:

| File | What it holds |
|---|---|
| `results.jsonl` | One row per question as soon as it is scored: the question, the expected answer, the model's answer and the score. |
| `traces.jsonl` | Everything behind each row: what was searched, the search's whole answer, the context the model read, and each model call's request, reply and token use. |
| `running.txt` | The score so far every 10 questions (`--every N` to change), overall and by category, with errors and spend. The same line is printed as the run goes. |
| `summary.json` | The same, rewritten each time, so it is current even for a run that stopped. |

While it runs, a status line every 30 seconds (`--status N` to change) says how many questions
are answered, how many app users are done, running or waiting, and what each running one is doing
and for how long - adding items, processing them, or asking:

```
[18:12:40] 6m30s · 2/5 answered · units 1 done, 4 running, 0 waiting · plus-1 asking 1/1 (4s) · plus-2 processing 648/665 items (5m40s) · ... · models $0.01
```

When the run finishes it prints the published number beside yours.

**Stopping and resuming.** Ctrl-C lets the questions already being asked finish and be written,
then stops. `--resume <run>` carries on: questions already scored are not asked again; an app
user that was part way through keeps its memory and is given only the items it does not hold yet.
If that memory holds anything a question still to be asked must not see, it is erased and built
again from the start. App users are erased only once all their questions are answered, so
`alexein-bench clean` removes those of a run you give up on.

## How a run works

Each benchmark is run the way an app would use Alexein, through the public
[Python client](https://pypi.org/project/alexein/):

1. **One app user per unit**: a LoCoMo conversation, a LongMemEval question's history, a RHELM
   character. Each starts empty and is erased when it is done.
2. **History added in time order.** Each item (a message, an email, a document) is added with
   its own date. A question is asked only once everything dated before it has been added and
   processed, and nothing from on or after its date is in memory yet. A question about last
   Tuesday sees only what was said before it was asked.
3. **One search per question.** The question goes to `search` as it is written. The memories
   that come back, rendered by the client's `context()`, are what the answering model reads,
   in the benchmark's own answer prompt.
4. **The benchmark's own scoring**: its judge prompt and model, or its exact-match metric,
   copied from the benchmark's authors. Each module names the source and version of what it
   copies.

Datasets are downloaded from their authors on first use, at a pinned version, and cached in
`~/.cache/alexein-bench`. Nothing from them is stored in this repository.

## Checking this repository

```sh
make check   # lint, types, tests (offline), and the public-text check
```

## Licence

The code is MIT. The datasets are their authors', under their own licences. The answer and judge
prompts each benchmark module copies are the benchmark authors', credited where they are copied.
