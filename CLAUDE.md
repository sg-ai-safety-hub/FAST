# FAST — Frontier AI Security Training

This repository holds the **content** of the FAST program: notebooks, exercises, briefs, slides,
datasets, and handouts. It is not an application codebase. Almost everything here is something a
participant reads, runs, or fills in.

## This repository is public

Everything committed here should be safe for anyone on the internet to read, including future
cohorts. That rule shapes what does and does not belong:

**Does not belong here.** Instructor-only hints, run-of-show and staffing notes, who teaches
what, speaker briefs and outreach, internal critique of the curriculum, budget and logistics,
anything about individual participants. That material lives in our internal workspace docs, not
in git. If a file would embarrass us or leak someone's information when read by a stranger, it
is in the wrong place.

**Does belong here, but held back from participants.** Lab solutions and exercise answer keys
live in this repo — they are part of the value participants take home, and they need them after
the week ends. They are simply not surfaced by default:

- `solution.ipynb` sits beside `lab.ipynb`, never inline in the notebook a participant works in.
- Participants are told plainly that struggling first is the point. This is an honour system and
  we say so rather than pretending it is enforced. The framing lives once in the root README.
- Never surface a solution as the obvious next click. No "→ solution" link at the top of a lab.

Teaching guidance that is fine to publish — what an exercise is designed to surface, its timing,
its prerequisites, the discussion it should provoke — is welcome, and helps anyone reusing this
material. Keep it participant-legible rather than writing it as private notes to ourselves.

## Program facts

- **Program**: FAST — Frontier AI Security Training. Website: https://www.securefast.ai/
- **Run by**: SASH (Singapore AI Safety Hub)
- **Dates**: 28 September – 2 October 2026, Singapore, co-located with SICW
- **Format**: 5 days, in person, fully funded, **20 participants per cohort**, drawn
  internationally with particular focus on Asia
- **Track**: Technical
- **Framing**: "attack and defend frontier models, agents, and the systems built around them"

## Who is in the room

Security engineers with real cybersecurity work behind them, and ML engineers/researchers who
have personally trained or fine-tuned models. **This is not an introductory programme.** Two
consequences for everything we write:

1. **Do not explain what a transformer is, what a gradient is, or what a threat model is.**
   Assume competence but link useful content in case. Explain the *specific* thing that is new (e.g. what a refusal direction
   is), not the surrounding fundamentals.
2. **The room is split.** A security engineer may have never run a training loop; an ML
   researcher may have never thought in terms of blast radius or insider threat. Content should
   bridge deliberately — the Day 2 "assume breach ↔ AI control" mapping is the template for
   this. When an exercise leans hard one way, say so in its README and give the other half a
   foothold.

## Delivery environment: Google Colab

Participants are granted accounts on **our Google Workspace** and run everything in **Google
Colab**. This is the single most important technical constraint in the repo. Every notebook must:

- **Run top-to-bottom on a fresh Colab runtime with no setup outside the notebook.** No local
  paths, no assumed clones, no environment variables the participant has to set. First cell
  installs, second cell imports, and it works.
- **Be frugal with compute.** Participants can run a *paid* GPU runtime, so we are not stuck on
  the free tier — but the budget is modest and shared across 20 people for five days. Design for
  a single mid-tier GPU (T4/L4 class, ~16–24GB) and treat anything above that as a cost we have
  to justify. Prefer **1–3B parameter models** (Llama/Gemma/Qwen small variants), short training
  runs, small batches, LoRA over full fine-tunes, quantized loads where they don't obscure the
  lesson. If an exercise genuinely needs more, say so in the notebook header with an estimated
  GPU-hour cost so we can decide deliberately.
- **Survive a disconnect.** Long-running cells are a failure mode: a participant who loses the
  runtime at minute 40 of a 60-minute exercise is lost for the session. Checkpoint to Drive, or
  structure the exercise so the expensive step is short and everything after it is cheap.
- **Be time-boxed to its slot.** An exercise budgeted at 1h needs to run in well under 1h of
  wall-clock compute — the rest of the hour is reading, discussion, and people getting stuck.
  State expected runtime per heavy cell.
- **Pin versions.** `pip install` without pins will break silently between now and September.
- **Gate downloads.** Hugging Face model pulls should be small, cached where possible, and not
  20 people hammering the same gated repo at once. Check licence/gating for every model we use.

Notebooks should also degrade gracefully: if a model download fails, the participant should get
a clear message, not a stack trace 200 lines down.

## Kinds of content in this repo

Not everything is a notebook. The curriculum mixes at least five formats, and each needs
different artifacts:

| Format | Example | What we ship |
| --- | --- | --- |
| Hands-on lab | Abliteration, backdoors & probes, output distributions | Colab notebook + separate solution notebook |
| Card / sorting exercise | AIS definitions | Printable card deck (PDF) + held-back answer key |
| Timeboxed hunt | System card easter-egg hunt | Question set, source docs, held-back answer key |
| Structured discussion / debate | Assume-breach mapping | Framing brief, prompt sheet, positions to argue |
| Team design exercise | AI control hackathon, wargaming | Scenario brief per team, materials list, plenary format |
| Worksheet | SMART goal, expectation exercise | Fillable template + how it gets used later in the week |
| Guest lecture | AISI, Apollo/Redwood, FLI/TamperSec/Lucid | Reading or context the session assumes |

Guest-lecture *speaker* briefs, and anything about who is teaching or supporting a session, stay
in the internal docs — only the participant-facing context lands here.

**Every exercise needs a short README**, notebook or not: the objective it serves, its format,
prerequisites, and what it's meant to surface. Leave duration out of a stub; the person leading
the session owns the timing and sets it later. Write it so a participant
or an outside reader could pick it up — not as private notes. Minute-by-minute run of show,
what to cut when running late, and instructor-only framing belong in the internal docs.

## Curriculum shape (draft — subject to change)

| Day | Theme | Core objective |
| --- | --- | --- |
| 0 | Setup & environment | Colab/GPU access verified, repo cloned, smoke test green |
| 1 | AI models | Shared terminology; where to find real model info (system cards, evals) |
| 2 | Control | Why alignment is hard; the AI control agenda; protocols and safety–usefulness tradeoffs |
| 3 | Open weight security | Why weights are harder to safeguard than an API: abliteration, backdoors, distillation |
| 4 | Verification | Racing dynamics; treaty verification; hardware-enabled mechanisms |
| 5 | Next steps | A concrete SMART goal, warm intros, expectation recap against Day 1 |

The authoritative curriculum source is **MASTERDOC v6 (Jul 2026)** plus the working curriculum
spreadsheet. This table is a convenience summary; when they disagree, they win. Objectives drive
exercises — if an exercise doesn't serve a stated training objective, say so rather than
quietly building it.

Two structural threads to preserve:
- **Day 1 expectation exercise → Day 5 closing recap.** The Day 1 answers get handed back on
  Day 5. Anything we build on Day 1 must produce artifacts that survive the week.
- **Day 3 is the highest cognitive load day.** Three hands-on labs back to back. Guard the
  pacing there aggressively.

## Handling dual-use content

Several Day 3 exercises are genuinely dual-use — removing refusal behavior from a model,
training trigger-conditioned backdoors, extracting capabilities via distillation. This is an
authorized professional security training program; that is exactly the context in which these
techniques should be taught, and we teach them properly. But:

- **Use toy and small open-weight models.** The pedagogical point lands on a 1–3B model. Scaling
  it up adds nothing educational and a lot of risk.
- **Do not commit weaponizable artifacts** — no abliterated checkpoints, no backdoored model
  weights, no curated harmful-prompt corpora in the repo. Notebooks generate what they need at
  runtime; datasets are referenced by source, not vendored.
- **Pair every offensive exercise with its defensive counterpart.** Backdoors ship with probe
  detection. Abliteration ships with the "so what does this mean for your weight security
  posture" discussion. This is the design principle, not a disclaimer.
- **Keep harmful/harmless prompt pairs minimal and clinical** — the smallest set that produces a
  measurable activation difference.

If a piece of content feels like it crosses from "understand the attack" into "here is a
deployable capability", flag it rather than shipping it.

## Writing conventions

- **Markdown** for prose. **Jupyter notebooks** (`.ipynb`) for labs. Slides: format TBD.
- Keep solutions in **separate files** from the work. Never leave an answer key inside a
  participant handout or a lab notebook.
- Every exercise file starts with a header block: **objective, format, prerequisites**. Duration
  is the session lead's to set, so a stub omits it; a built lab may still state its real runtime.
- Timings in the curriculum are real budgets (30min, 45min, 1h, 1.5h). Build to the budget and
  state where the slack is.
- Cite sources with real links. This audience will check them, and half the exercises
  (system cards, GDM control roadmap, sleeper agents, RAND SL1–SL5, AI 2040, INF Treaty) are
  built directly on public documents.
- **Watch US-centric framing**, especially on Day 4. The cohort is international with an Asia
  focus. AI 2040 and the INF case study are useful *tools to think with*, not the world's
  default frame — caveat them explicitly in the material itself.
- Neutral pronouns (they/them) for participants, speakers, and anyone whose pronouns we don't
  know.

## Repository layout

```
README.md              course homepage — schedule, setup, index
ruff.toml              lint config for the whole repo (tools, days, package)
tools/                 build_labs.py, run_notebooks.py — repo build scripts (run from root)
assets/                shared images
src/                   everything the curriculum is built from
  packages/fast/       the installable `fast` package — pip-installed into Colab from this repo
    pyproject.toml     package + dev deps live here, not at the repo root
    fast/              shared lab utilities: colab, models, testing, labs/
  examples/template/   worked example of every lab mechanism — copy to start a new lab
  dayN-<theme>/
    README.md          objectives + run of show (the authoritative exercise order)
    <exercise>/
      README.md        objective, format, prerequisites
      lab.py           source of truth for notebooks — edit this
      lab.ipynb        generated
      solution.ipynb   generated
```

**Exercise directories are unnumbered on purpose.** Ordering lives in the day README, so
reshuffling the run of show is a one-line edit rather than a rename that breaks every Colab
link people have bookmarked. Days are numbered because they don't move.

Guest lectures and 1:1 sessions get no directory — speaker briefs are internal. Only the reading
a session assumes lands here, inside the relevant exercise.

## Building labs

Notebooks are **generated, never hand-written**. Each lab is authored once as a jupytext
percent-format `.py`; `tools/build_labs.py` renders both the participant and solution notebooks
from it, so they cannot drift. Editing a `.ipynb` directly means losing the edit on next build.

```sh
uv venv && uv pip install -e "src/packages/fast[dev]"   # contained env in .venv/ — never a global python
uv run python tools/build_labs.py         # rebuild
uv run python tools/build_labs.py --check # verify nothing is stale
```

Functions participants implement are decorated with `@fast.testing.exercise` — a runtime no-op
that the build finds in the AST:

```python
@exercise
def difference_in_means(harmful, harmless):
    """Return the unit-norm direction separating the two activation sets."""
    direction = harmful.mean(axis=0) - harmless.mean(axis=0)
    return direction / np.linalg.norm(direction)
```

The participant notebook keeps the decorator, signature and docstring, and gets
`raise NotImplementedError` for a body. **The docstring is the spec** — the body doesn't survive,
so everything a participant needs must be in the signature and the docstring. `@exercise(stub=…)`
leaves a scaffold instead; reach for it rarely, since a scaffold that pre-shapes the answer
removes the part worth doing.

Bodies are located by parsing, not by scanning for comment markers, so indentation and nesting
are the parser's problem rather than ours. It also means **exercises must be functions** — which
is what you want anyway, since a check needs something to call.

Whole cells are marked by a comment on their first line, since a cell has nothing to decorate:
`# @solution-only` drops it from the lab, `# @lab-only` drops it from the solution.

Cell IDs are derived from content hashes so rebuilds don't produce spurious diffs. Outputs are
stripped from generated notebooks; when a lab is finished, run the solution once and commit it
with outputs intact, so it can be read without a GPU.

`src/examples/template/` is a worked, CI-executed example of every mechanism. Start a lab by copying
it and `fast/labs/examples/template.py`.

## Checks

Participants need to know whether their implementation is right without asking an instructor.
Each lab gets a module under `fast/labs/`, named and placed to mirror the lab directory
(`src/day1-models/output-distributions/` -> `fast.labs.day1_models.output_distributions`; hyphens
become underscores), holding **both** its shared fixtures and
its checks, so a notebook needs one import:

```python
from fast.labs.day1_models import output_distributions as lab

harmful, harmless = lab.activation_pairs()
lab.check_difference_in_means(difference_in_means)
```

Checks ship inside the installed package rather than in the notebook, so they're available in
Colab without sitting where they'd give the answer away. They are still readable in this repo —
same honour system as solutions.

- **Assert properties, not reference implementations.** `require(is_unit_norm(fn(x)))` doesn't
  hand over the answer; `require(np.allclose(fn(x), reference(x)))` does.
- **Failure messages are read by someone stuck.** Say what was expected and what was seen.
- `require()` for each property, `@checker("name")` to wrap. Printing is for the participant;
  the raise is for CI.
- **A check must fail on the shipped stub.** One that passes on both `lab.ipynb` and
  `solution.ipynb` is testing nothing.

## CI

`.github/workflows/ci.yml` runs three things on every push: ruff, `build_labs.py --check` (that
notebooks match their sources), and `run_notebooks.py` (executes every notebook except
`lab.ipynb`, which is deliberately incomplete). A lab whose checks disagree with its own
reference implementation fails there rather than in the room.

The runners have no GPU, and CPU `torch` is a dev dependency for exactly this reason. Labs that
train should scale themselves down rather than opt out:

```python
steps = 10 if ci_mode() else 500
```

A lab excluded from CI is a lab nobody notices is broken until the morning it runs. If one has to
be excluded, say so out loud rather than letting it quietly not run.

The same fast checks (ruff and `build_labs.py --check`) run locally through
`.pre-commit-config.yaml` — enable them once with `uv run pre-commit install`. Notebook execution
is CI-only; it's too slow for a commit hook. The hooks are local, so they use the tools already
in `.venv` and match CI exactly.

## Environment

`uv` with a `.venv/` inside the repo. There is no system Python here — always `uv run` or
`.venv/bin/python`, never a global interpreter.

`src/packages/fast/pyproject.toml` deliberately does **not** declare `torch`. Colab already ships a CUDA build;
declaring it risks pip swapping in a CPU wheel mid-install and breaking a lab.

The install cell in every notebook is identical, pip-installing
`git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast` — so
changing the source is one
find/replace, not an archaeology exercise. The repo is **public**, so the clone needs no
authentication. Pin `@main` to a release tag before the program runs.

## Status

Roughly eight weeks out from delivery (today: August 2026). The curriculum draft is explicitly
"very much subject to change" — objectives are firmer than exercises, and exercises are firmer
than timings. Expect churn; keep content modular enough that dropping one exercise doesn't break
the day around it.
