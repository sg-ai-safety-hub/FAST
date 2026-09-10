# Day 0 — Setup & Environment

Done before Day 1 starts, not during it.

1. Sign in to Colab with the Google account you were issued for the program.
2. Open [`smoke_test.ipynb`](smoke_test.ipynb) in Colab.
3. Set the runtime to a T4 GPU: **Runtime > Change runtime type > T4 GPU**.
4. Run every cell. It takes about two minutes.

If a cell fails, bring the error to setup. An environment problem found on Day 0 is a
non-event; the same problem found on Wednesday morning costs you a lab.

## How a lab works

Every lab is one `lab.ipynb` in an exercise directory, and every one is laid out the same way.

**It installs itself.** The first cell pulls in the shared `fast` package; the second calls
`setup()`, which seeds everything and tells you what GPU you're on. Nothing to install locally,
nothing to clone.

**You fill in functions.** Anything marked `@exercise` is yours to implement. The signature and
the docstring are the specification — the docstring says exactly what to return:

```python
@exercise
def difference_in_means(harmful, harmless):
    """Return the unit-norm direction separating the two activation sets."""
    raise NotImplementedError
```

**The lab tells you if you're right.** After each exercise there's a check. Run it as often as
you like:

```python
lab.check_difference_in_means(difference_in_means)
# ✅ difference_in_means — 4 checks passed
```

A failing check says what it expected and what it got, so most of the time you can fix it
without waiting for anyone:

```
❌ difference_in_means — direction should be unit norm, got 2.8431
```

**Shared data comes from the same place as the checks.** `lab.activation_pairs()` and friends
give you the fixtures an exercise needs, so you spend your time on the idea rather than on
plumbing.

## Solutions

Each lab has a `solution.ipynb` next to it. They're worth more to you unopened — getting stuck
and staying stuck for a bit is where most of the learning happens, and these are built so the
struggle is the point rather than an obstacle to it. Nothing is locked and nobody is checking.

Reasonable to open one: you're genuinely blocked and no instructor is free, you've finished and
want to compare, or the week is over and you're coming back to the material. Solutions ship with
their outputs saved, so you can read one without a GPU.

## Things that will bite you

**Colab disconnects.** Anything expensive checkpoints to your Drive — that's what the
`mount_drive()` cell is for. Don't skip it and then lose forty minutes of training.

**Watch the compute.** Every lab states its expected runtime at the top. If a cell is taking far
longer than advertised, something is wrong — usually the wrong runtime type. Check the GPU line
that `setup()` printed rather than waiting it out.
