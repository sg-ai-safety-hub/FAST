# Contributing

## Setup and workflow

Notebooks are generated from a jupytext `.py` file next to them. Edit `lab.py`, never the
`.ipynb`, then rebuild. Setup and commands are in the README under
[Working on this repo](README.md#working-on-this-repo). Before opening a PR:

```sh
uv run python tools/build_labs.py           # regenerate lab.ipynb and solution.ipynb
uv run python tools/build_labs.py --check   # CI fails if these are stale
```

CI also runs every notebook, so a change that breaks a solution or makes a check pass on the
empty stub shows up there.

## Writing for a lab

- **Don't give the answer away.** The exercise is the part worth doing. Docstrings, prose and
  worked examples should say *what* a function must do it's built around. If an explanation
  spoils an exercise, put it after the check, where it explains what just happened.
- **Keep the main path light.** Participants should come away with a feel for the idea, not
  every derivation. Anything that helps someone get through the notebook stays inline; optional
  depth (maths, edge cases, background) goes in a collapsible block:

  ```markdown
  <details>
  <summary><b>Optional: what the scorer computes</b></summary>

  Leave a blank line after the summary so markdown and LaTeX render.

  </details>
  ```

- **Checks test properties, not a reference answer**, and must fail on the shipped stub.

The full conventions are in [CLAUDE.md](CLAUDE.md).

## This repository is public

Don't commit model weights, harmful-prompt datasets, or anything about individual participants.
Labs must generate what they need at runtime as much as possible.
