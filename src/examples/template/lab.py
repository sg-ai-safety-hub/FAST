# %% [markdown]
# # Example lab — how a FAST lab is put together
#
# A worked, runnable example of every mechanism a lab has: solution markers, shared
# fixtures, and checks participants can run themselves. Copy `examples/template/lab.py`
# to `dayN-<theme>/<exercise>/lab.py` and `fast/labs/examples/template.py` to
# `fast/labs/<dayN_theme>/<exercise>.py` to start a real one.
#
# The maths here is a stripped-down version of the abliteration lab — find the direction
# separating two sets of activations, then project it out — but on synthetic numpy data so
# it needs no GPU and runs in CI.
#
# **Budget:** state real wall-clock compute here, e.g. "~6 min on a T4". The run of show
# depends on it.

# %%
# !pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast

# %%
import numpy as np

from fast.labs.examples import template as lab
from fast.testing import exercise

# %% [markdown]
# ## Exercise 1 — find the separating direction
#
# `activation_pairs()` gives you activations from paired prompts: one set that triggers the
# behaviour you care about, one that doesn't. The difference in their means points along the
# axis that distinguishes them.
#
# Return it normalised, pointing towards `harmful`.


# %%
@exercise
def difference_in_means(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """Return the unit-norm direction separating the two activation sets.

    The docstring survives into the participant notebook and the body doesn't, so it
    carries the whole specification. Write it as one.
    """
    direction = harmful.mean(axis=0) - harmless.mean(axis=0)
    return direction / np.linalg.norm(direction)


lab.check_difference_in_means(difference_in_means)

# %% [markdown]
# ## Exercise 2 — project it out
#
# Remove the direction from a set of activations, leaving everything orthogonal to it
# untouched. This is the operation that, applied to weights rather than activations, ablates
# a behaviour.
#
# `stub=` leaves a scaffold instead of `raise NotImplementedError`. Reach for it rarely — a
# scaffold that pre-shapes the answer removes the part worth doing.


# %%
@exercise(stub="return activations - ...")
def project_out(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove `direction` from every row of `activations`, leaving the rest untouched."""
    return activations - np.outer(activations @ direction, direction)


lab.check_project_out(project_out)

# %% [markdown]
# ## Cell-level markers

# %%
# @solution-only
# Only in solution.ipynb — use it for a plot of the real answer, or commentary that would
# give the exercise away.
harmful, harmless = lab.activation_pairs()
direction = difference_in_means(harmful, harmless)
print(f"recovered direction, largest component at index {direction.argmax()}")

# %%
# @lab-only
# Only in lab.ipynb — use it for a hint, or a cell participants fill in freehand where
# there's no single right answer.
print("Stuck? The two sets differ along exactly one axis.")
