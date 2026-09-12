# FAST — Frontier AI Security Training

Course materials for [FAST](https://www.securefast.ai/), run by SASH (Singapore AI Safety Hub).
Technical track, 28 September to 2 October 2026, Singapore.

Five days on attacking and defending frontier models, agents, and the systems around them. The
labs, their solutions, and the utilities behind them are all here, public and reusable.

## Start here

Do [`day0-setup/`](src/day0-setup/) before Day 1, not on Wednesday morning. It takes two minutes
and confirms your Colab account works. Doing it early is the whole point of it.

## The week

| Day | | |
| --- | --- | --- |
| 0 | [Setup & Environment](src/day0-setup/) | Colab, GPU, smoke test |
| 1 | [AI Models](src/day1-models/) | Output distributions, instruction hierarchies |
| 2 | [Control](src/day2-control/) | Control hackathon |
| 3 | [Open Weight Security](src/day3-weights/) | Abliteration, backdoors, distillation |

The programme itself runs five days; this repo holds the hands-on material — the labs and the
control hackathon. Each day's README lists that day in running order.

## Labs

Labs run in Google Colab on the account you were issued. Open `lab.ipynb` in any lab directory
and run the first cell. It installs everything, so there's nothing to set up locally.

Labs check themselves. After each function you write, a cell tells you whether it's right:

```python
lab.check_next_token_probs(next_token_probs)
# ✅ next_token_probs — 6 checks passed
```

A failing check says what it expected and what it saw, so you can usually fix it without waiting
for an instructor.

Every lab has a `solution.ipynb` next to it, and it's worth more to you unopened. Getting stuck
and staying stuck for a while is where most of the learning happens, and the labs are built so
that struggle is the work rather than something in its way. Nothing is locked and nobody checks.
Open one when you're truly blocked and no instructor is free, when you've finished and want to
compare, or when the week is over and you're reviewing. Solutions keep their saved outputs, so
you can read one without a GPU.

## Working on this repo

Notebooks are generated, not written by hand. Each lab is authored once as a jupytext
percent-format `.py` file, and `tools/build_labs.py` renders `lab.ipynb` and `solution.ipynb`
from it, so the two can't drift. Edit the `.py`, never the `.ipynb`.

```sh
uv venv
uv pip install -e "src/packages/fast[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
uv run pre-commit install                 # enable the local commit checks
uv run python tools/build_labs.py         # rebuild notebooks
uv run python tools/build_labs.py --check # check nothing is stale
uv run python tools/run_notebooks.py      # run every notebook except lab.ipynb
```

Before each commit, the hooks run ruff and the notebook sync check on what you're committing.
The full notebook execution is CI's job, since it downloads models and takes minutes.

Shared helpers, fixtures, and checks live in [`src/packages/fast/`](src/packages/fast/) and
install into Colab straight from this repo. [`src/examples/template/`](src/examples/template/) is
a worked example of every lab mechanism; copy it to start a new one. CI runs lint, the staleness
check, and every notebook on each push. See [CLAUDE.md](CLAUDE.md) for conventions.

## Licence

Course content (prose, exercises, worksheets, slides) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code under MIT. See [LICENSE](LICENSE).
