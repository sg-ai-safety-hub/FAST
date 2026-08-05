# FAST — Frontier AI Security Training

Course materials for [FAST](https://www.securefast.ai/), run by SASH (Singapore AI Safety Hub).
Technical track, 28 September – 2 October 2026, Singapore.

Five days on attacking and defending frontier models, agents, and the systems built around them.
Everything here is public and reusable — labs, exercises, worksheets, and solutions.

## Start here

Before Day 1, work through [`day0-setup/`](day0-setup/). It takes two minutes and confirms your
Colab environment works. Doing it on Day 0 rather than Wednesday morning is the whole point.

## The week

| Day | | |
| --- | --- | --- |
| 0 | [Setup & Environment](day0-setup/) | Colab, GPU, smoke test |
| 1 | [AI Models](day1-models/) | Terminology, system cards, evals, logprobs |
| 2 | [Control](day2-control/) | Alignment, assume breach, protocols, defer-to-resample |
| 3 | [Open Weight Security](day3-weights/) | Threat tiering, abliteration, backdoors, distillation |
| 4 | [Verification](day4-verification/) | Racing dynamics, INF Treaty, hardware mechanisms |
| 5 | [Next Steps](day5-next-steps/) | Where to go next, SMART goals, closing |

Each day's README lists its exercises in running order.

## Labs

Labs run in Google Colab on the account you were issued. Open `lab.ipynb` from any exercise
directory — the first cell installs everything it needs, so there is nothing to set up locally.

Labs check themselves. After each function you write, a cell tells you whether it's right:

```python
lab.check_difference_in_means(difference_in_means)
# ✅ difference_in_means — 4 checks passed
```

A failing check says what was expected and what it saw, so you can usually fix it without
waiting for an instructor.

Each lab ships with a `solution.ipynb` beside it. **They're worth more to you unopened.** Getting
stuck and staying stuck for a while is where most of the learning happens, and the labs are built
so the struggle is the point rather than an obstacle to it. Nothing is locked and nobody is
checking — it's simply your call.

Good reasons to open one anyway: you're genuinely blocked and no instructor is free, you've
finished and want to compare approaches, or the week is over and you're revisiting the material.
That last case is why solutions ship with their outputs intact — you can read one without a GPU.

## Working on this repo

Notebooks are generated, not hand-written. Each lab is authored once as a jupytext
percent-format `.py` file; `tools/build_labs.py` renders `lab.ipynb` and `solution.ipynb` from
it, so the two can't drift apart. Edit the `.py`, never the `.ipynb`.

```sh
uv venv
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
uv run python tools/build_labs.py         # rebuild notebooks
uv run python tools/build_labs.py --check # verify nothing is stale
uv run python tools/run_notebooks.py      # execute every notebook but lab.ipynb
```

Shared helpers, fixtures and checks live in [`fast/`](fast/) and install into Colab straight from
this repo. [`examples/template/`](examples/template/) is a worked example of every mechanism a lab
has — copy it to start a new one. CI runs lint, the staleness check, and every notebook on each
push. See [CLAUDE.md](CLAUDE.md) for conventions.

## Licence

Course content (prose, exercises, worksheets, slides) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code under MIT. See [LICENSE](LICENSE).
