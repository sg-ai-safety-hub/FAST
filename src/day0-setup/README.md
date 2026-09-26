# Day 0: Setup & Environment

Do this before Day 1 starts, not during it.

1. Sign in to Colab with the Google account you were issued for the program.
2. Open [`smoke_test.ipynb`](smoke_test.ipynb) in Colab.
3. Set the runtime to a T4 GPU: **Runtime > Change runtime type > T4 GPU**.
4. Run every cell. It takes about two minutes.

If a cell fails, bring the error to setup. An environment problem found on Day 0 is a non-event.
The same problem on Wednesday morning costs you a lab.

## How a lab works

Every lab is one `lab.ipynb` in a lab directory, and they're all laid out the same way.

**It installs itself.** The first cell pulls in the shared `fast` package. The second calls
`setup()`, which seeds the run and prints the GPU you're on. Nothing to install locally, nothing
to clone.

**You fill in functions.** Anything marked `@exercise` is yours to implement. The signature and
docstring are the specification, and the docstring says exactly what to return:

```python
@exercise
def next_token_probs(logits, temperature=1.0):
    """Turn a 1D tensor of logits into a probability distribution over the vocabulary."""
    raise NotImplementedError
```

**The lab tells you if you're right.** Each exercise is followed by a check you can run as often
as you like:

```python
lab.check_next_token_probs(next_token_probs)
# ✅ next_token_probs: 6 checks passed
```

A failing check says what it expected and what it got, so most of the time you can fix it without
waiting for anyone:

```
❌ next_token_probs: temperature is backwards: low temperature should concentrate mass
```

**Shared data comes from the same import as the checks.** Calls like `lab.logits_for(...)` and
`lab.mc_items()` hand you the fixtures an exercise needs, so your time goes on the idea rather
than the plumbing.

## Solutions

Each lab has a `solution.ipynb` beside it, and it's worth more to you unopened. Getting stuck and
staying stuck for a bit is where most of the learning happens, and the labs are built so that
struggle is the work. Nothing is locked and nobody checks.

Open one when you're truly blocked and no instructor is free, when you've finished and want to
compare, or when the week is over and you're coming back to the material. Solutions keep their
saved outputs, so you can read one without a GPU.

## Things that will bite you

**Colab disconnects.** Anything expensive checkpoints to your Drive, which is what the
`mount_drive()` cell is for. Skip it and a dropped runtime can cost you forty minutes of training.

**Watch the compute.** Every lab states its expected runtime at the top. A cell running far
longer than advertised usually means the wrong runtime type. Check the GPU line `setup()` printed
instead of waiting it out.
