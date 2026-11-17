<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/alexein-dark.svg">
    <img src="assets/alexein-light.svg" alt="Alexein" width="88" height="88">
  </picture>
</p>

<h1 align="center">Alexein benchmarks</h1>

<p align="center">
  The numbers on <a href="https://alexein.ai/benchmarks">alexein.ai/benchmarks</a>, and the code to run each one yourself<br>
  against Alexein's public API, with one command and your own keys.
</p>

<p align="center">
  <a href="LICENSE"><img alt="Licence: MIT" src="https://img.shields.io/badge/licence-MIT-e4e4e1?style=flat-square&labelColor=1d1d20"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-e4e4e1?style=flat-square&labelColor=1d1d20">
  <a href="https://docs.astral.sh/uv/"><img alt="Runs with uv" src="https://img.shields.io/badge/runs%20with-uv-e4e4e1?style=flat-square&labelColor=1d1d20"></a>
  <a href="https://alexein.ai/benchmarks"><img alt="alexein.ai/benchmarks" src="https://img.shields.io/badge/alexein.ai-benchmarks-e4e4e1?style=flat-square&labelColor=1d1d20"></a>
</p>

<p align="center">
  <a href="#at-a-glance">At a glance</a> ·
  <a href="#results">Results</a> ·
  <a href="#benchmark-by-benchmark">Benchmark by benchmark</a> ·
  <a href="#run-one">Run one</a> ·
  <a href="#what-a-run-writes">What a run writes</a> ·
  <a href="#how-a-run-works">How a run works</a>
</p>

<br>

## At a glance

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/glance-dark.svg">
  <img src="assets/glance-light.svg" alt="Four headline results. Plus 58.0 points: GPT-4.1-mini on MemoryAgentBench fact consolidation, 36.0 reading the whole history, 94.0 with Alexein. Number 1 memory framework on Microsoft's RHELM, ahead of memU, Mem0 and MemGPT. 3.2 times the LoCoMo-Plus paper's best memory: 55.4 against A-Mem's 17.20. 70.57% pass rate on BEAM's million-token conversations, level with Mem0's 70.14%.">
</picture>

<br><br>

The clearest test of a memory is to hold everything else still. Same questions, same scoring,
same answer model. The only thing that changes is whether the model reads the whole history or
reads Alexein:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/same-model-dark.svg">
  <img src="assets/same-model-light.svg" alt="GPT-4.1-mini reading the whole history, then reading Alexein. MemoryAgentBench, facts that keep changing: 36.0 to 94.0. RHELM, a year of chats, emails and documents: 33.9 to 48.5. LoCoMo: 80.6 to 94.6. MemoryAgentBench, a whole detective novel: 56.3 to 66.2. MemoryAgentBench, one fact in a long history: 83.0 to 92.0.">
</picture>

## Results

Every number below is written exactly as the website shows it, and every row runs with one
command, written out under its benchmark below (or run `alexein-bench list`). Alexein's results
are self-reported; the other systems' results in the charts are as their own sources publish
them, each linked under its chart.

<!-- results:start -->
| Benchmark | Alexein | Questions | Answers / grades | Run it |
|---|---:|---:|---|---|
| [RHELM](https://github.com/microsoft/RHELM)<br>Chats, emails and documents | **48.5** | 1,305 | gpt-4.1-mini / gpt-4.1-mini | [Command](#rhelm) |
| [RHELM](https://github.com/microsoft/RHELM)<br>Chats only | **35.6** | 1,305 | gpt-4.1-mini / gpt-4.1-mini | [Command](#rhelm) |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257)<br>Fact consolidation (single-hop) | **94.0** | 100 | gpt-4.1-mini / no model (exact match) | [Command](#memoryagentbench) |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257)<br>RULER-QA1, accurate retrieval | **92.0** | 100 | gpt-4.1-mini / no model (exact match) | [Command](#memoryagentbench) |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257)<br>DetectiveQA, all 10 books | **66.2** | 71 | gpt-4.1-mini / no model (exact match) | [Command](#memoryagentbench) |
| [LoCoMo](https://github.com/snap-research/locomo)<br>All questions, categories 1-4 | **94.6** | 1,540 | gpt-4.1-mini / gpt-4o-mini, majority of three | [Command](#locomo) |
| [LoCoMo-Plus](https://arxiv.org/abs/2602.10715)<br>All questions | **55.4** | 401 | gpt-4.1-mini / the paper's judge | [Command](#locomo-plus) |
| [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)<br>All questions | **91.6** | 500 | gemini-3.6-flash / gpt-4o-mini | [Command](#longmemeval-s) |
| [BEAM, 1M-token split](https://arxiv.org/abs/2510.27246)<br>Mean score | **63.49** | 700 | gpt-5 / gpt-5 | [Command](#beam) |
| [BEAM, 1M-token split](https://arxiv.org/abs/2510.27246)<br>Pass rate | **70.57%** | 700 | gpt-5 / gpt-5 | [Command](#beam) |
<!-- results:end -->

Each command runs the same dataset, questions, prompts, models and scoring as the published
setup. Expect a score close to the published one, not the same digits. Some benchmarks sample
their models, as RHELM does, so every run is one draw. And Alexein keeps improving, so a run
today reads the memory of today.

## Benchmark by benchmark

Every chart draws each bar from 0 on the same 0 to 100 scale, so a small gap looks as small as it
is. Alexein is in ink, other memory systems in grey, and anything with no memory system, a model
reading the whole history or plain retrieval, as a pale bar.

### RHELM

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/rhelm-dark.svg">
  <img src="assets/rhelm-light.svg" alt="RHELM, 1,305 questions. Chats, emails and documents: GPT-5.5 with a 1M-token context 73.3, Claude Opus 4.6 with a 1M-token context 69.1, Alexein 48.5, Qwen3.5-397B-A17B 44.3, Claude Opus 4.5 retrieval 38.1, GPT-4.1-mini with a 1M-token context 33.9, memU 33.6, Mem0 28.9. Chats only: GPT-5.5 57.0, Claude Opus 4.6 49.7, Claude Opus 4.5 retrieval 36.2, Alexein 35.6, Qwen3.5-397B-A17B 34.6, memU 23.1, Mem0 21.1, MemGPT 13.9.">
</picture>

Sources: [RHELM leaderboard](https://github.com/microsoft/RHELM) ·
[RHELM, arXiv 2605.31086](https://arxiv.org/abs/2605.31086)

```sh
alexein-bench run rhelm --variant with      # chats, emails and documents
alexein-bench run rhelm --variant without   # chats only
```

### MemoryAgentBench

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/memoryagentbench-dark.svg">
  <img src="assets/memoryagentbench-light.svg" alt="MemoryAgentBench, fact consolidation, single-hop. Alexein 94.0 with GPT-4.1-mini, MindMemOS with Dreaming 92.0, GPT-4o reading the whole context 60.0, HippoRAG 2 54.0, BM25 48.0, Claude 3.7 Sonnet whole context 43.0, GPT-4.1-mini whole context 36.0, Gemini 2.0 Flash whole context 30.0, MemGPT 28.0, Cognee 28.0, MemoRAG 21.0, Mem0 18.0, MIRIX 14.0, GraphRAG 14.0, Zep 7.0.">
</picture>

Sources: [MemoryAgentBench, arXiv 2507.05257](https://arxiv.org/abs/2507.05257) ·
[MindMemOS, arXiv 2608.12428](https://arxiv.org/abs/2608.12428). The same model reading the whole
history is in the chart [at the top](#at-a-glance), for all three sets.

```sh
alexein-bench run mab --variant fc          # fact consolidation, single-hop
alexein-bench run mab --variant ruler       # RULER-QA1, accurate retrieval
alexein-bench run mab --variant detective   # DetectiveQA, all 10 books
```

### LoCoMo

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/locomo-dark.svg">
  <img src="assets/locomo-light.svg" alt="LoCoMo, 1,540 questions, every row answering with GPT-4.1-mini. Alexein 94.6, MindMemOS 94.03, Mnemis 93.9, EverOS 93.05, EMem-G 85.3, Zep 85.2, MemOS 80.8, the whole conversation with no memory 80.6, Nemori 79.5, RAG 73.8, LangMem 73.4, memU 66.7, Mem0 66.3, PREMem 65.8.">
</picture>

Sources: [Mnemis, arXiv 2602.15313](https://arxiv.org/abs/2602.15313) ·
[MindMemOS, arXiv 2608.12428](https://arxiv.org/abs/2608.12428). LoCoMo's answer key has known
errors, listed in an [audit of it](https://github.com/dial481/locomo-audit); they affect every
top system's score.

```sh
alexein-bench run locomo --variant all
```

### LoCoMo-Plus

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/locomo-plus-dark.svg">
  <img src="assets/locomo-plus-light.svg" alt="LoCoMo-Plus, 401 questions. Alexein 55.4 with GPT-4.1-mini. The paper's Table 1: Gemini 2.5 Pro reading the whole conversation 26.06, Gemini 2.5 Flash 24.67, GPT-4o 21.05, GPT-4.1 18.63, A-Mem 17.20, Mem0 15.80, RAG 15.55, SeCom 14.90, gpt-5-nano 14.84.">
</picture>

Source: [LoCoMo-Plus, arXiv 2602.10715](https://arxiv.org/abs/2602.10715). This harness gives
each LoCoMo conversation one app user holding all the cues placed in it. A later paper reports a
higher score from its own runs.

```sh
alexein-bench run locomo_plus --variant all
```

### LongMemEval-S

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/longmemeval-dark.svg">
  <img src="assets/longmemeval-light.svg" alt="LongMemEval-S, 500 questions. Mem0 Platform 93.4, as Mem0's blog reports it. Alexein 91.6. Mem0 OSS 91.0, run in the same setup.">
</picture>

Source: [Mem0's blog](https://mem0.ai/blog/the-token-efficient-memory-algorithm-now-has-temporal-reasoning),
for Mem0 Platform's figure. The benchmark:
[LongMemEval](https://github.com/xiaowu0162/LongMemEval). Its published setup answers with a
model other than OpenAI's, so this one needs an OpenRouter key.

```sh
alexein-bench run longmemeval --variant all
```

### BEAM

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/beam-dark.svg">
  <img src="assets/beam-light.svg" alt="BEAM, 1M-token split, 700 questions, GPT-5 answering and judging in Mem0's published harness. Mean score: Mem0 64.09, Alexein 63.49. Pass rate: Alexein 70.57%, Mem0 70.14%.">
</picture>

Sources: [Mem0's published BEAM harness](https://github.com/mem0ai/memory-benchmarks) ·
[BEAM, arXiv 2510.27246](https://arxiv.org/abs/2510.27246).
[Hindsight](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota) and
[Exabase](https://exabase.io/research/exabase-achieves-state-of-the-art-on-beam-benchmark) report
BEAM accuracy from their own setups; the only like-for-like comparison here is the one with Mem0.

```sh
alexein-bench run beam --variant 1m         # the mean score and the pass rate, in one run
```

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
models will cost on your key, then asks you to confirm. When it finishes, it prints the published
number beside yours.

```sh
alexein-bench list                                  # every number, and the command that runs it
alexein-bench run rhelm --variant with --limit 20   # a quick check on 20 questions
alexein-bench run rhelm --resume 20261117a1b2c3     # carry on a run that stopped
alexein-bench clean                                 # erase app users a stopped run left
```

From a clone: `uv run alexein-bench ...`.

## What a run writes

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

**Stopping and resuming.** Ctrl-C lets the questions already being asked finish and be written,
then stops. `--resume <run>` carries on: questions already scored are not asked again; an app
user that was part way through keeps its memory and is given only the items it does not hold yet.
If that memory holds anything a question still to be asked must not see, it is erased and built
again from the start. App users are erased only once all their questions are answered, so
`alexein-bench clean` removes those of a run you give up on.

## How a run works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/how-a-run-works-dark.svg">
  <img src="assets/how-a-run-works-light.svg" alt="A week of items on a timeline, Monday to Sunday. A question is asked on Thursday afternoon: everything dated before it has been added and processed; everything dated after it is added only once it is answered.">
</picture>

<br>

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
make check    # lint, types, tests (offline), and the public-text check
make readme   # redraw the README's table and charts from results/
```

The README's table and charts are drawn from `results/*.json`, and the other systems' figures
from [`assets/others.json`](assets/others.json), so neither can show a number the results don't
hold. `make check` fails if either is out of date, or if a chart's title claims something its
numbers no longer support.

## Licence

The code is MIT. The datasets are their authors', under their own licences. The answer and judge
prompts each benchmark module copies are the benchmark authors', credited where they are copied.
The Alexein name and mark are Alexein's.
