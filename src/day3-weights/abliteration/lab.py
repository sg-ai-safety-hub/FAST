# %% [markdown]
# # Abliteration — removing refusal from the weights
#
# A safety-tuned model refuses some requests. That refusal is a behaviour the weights learned,
# and like most learned behaviours it turns out to be carried by a single direction in the
# model's residual stream: run a batch of prompts the model refuses and a batch it answers, and
# the two clouds of activations sit on opposite sides of one axis
# ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)).
#
# If refusal lives along one direction, you can take it out. This lab finds that direction from a
# small set of prompts, removes it from the model's activations while it generates, and then bakes
# the same removal into the weights so it holds with nothing attached. The last step is the one
# that matters for weight security: it turns a shipped checkpoint into one that no longer refuses,
# and there is no way to put it back for someone who already has the file.
#
# The maths is the difference in means and a projection, the same two operations as the worked
# template, now on a real model.
#
# **Duration:** 75 min. **Prerequisites:** Day 0. **GPU:** recommended, a 1.5B model fits a T4.
# One pass over the model to find the direction, then linear algebra. Nothing trains.
#
# The technique is dual-use, so the lab stays on a small open model and a short set of prompts,
# and Part 4 is the other half of the story: what it means for weight security that refusal comes
# off this cheaply.

# %%
import os

token = os.environ.get("GITHUB_TOKEN")
if not token:
    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
    except Exception:  # noqa: BLE001 — not on Colab, or the secret isn't set
        token = None
auth = f"{token}@" if token else ""
# !pip install -q git+https://{auth}github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast

# %%
import torch

from fast.colab import setup
from fast.labs.day3_weights import abliteration as lab
from fast.testing import exercise

setup(require_gpu=False)
model, tokenizer = lab.load()

# %% [markdown]
# ## Part 1 — find the refusal direction
#
# The recipe is the difference in means. Take the residual-stream activations for prompts the
# model refuses and for prompts it answers, average each set into one vector, and subtract. What's
# left points along the axis that separates "about to refuse" from "about to answer". Everything
# the two sets have in common cancels; the refusal is what remains.
#
# `lab.collect_residuals` runs each prompt through the chat template and reads the hidden state
# above the last token, the position that decides the first token of the reply. It returns one
# vector per prompt, so two stacks of shape `(n_prompts, d_model)`.

# %%
harmful_acts = lab.collect_residuals(model, tokenizer, lab.harmful_prompts())
harmless_acts = lab.collect_residuals(model, tokenizer, lab.harmless_prompts())
print(f"harmful  activations: shape {tuple(harmful_acts.shape)}  (n_prompts, d_model)")
print(f"harmless activations: shape {tuple(harmless_acts.shape)}  (same shape, matched prompts)")


# %% [markdown]
# ### Exercise: the refusal direction
#
# Average each set over its prompts and subtract, then normalise to unit length so it names a
# direction and not a magnitude. Point it from harmless toward harmful, so that adding it pushes
# the model to refuse and removing it pushes it to answer. Later steps rely on that sign.


# %%
@exercise
def refusal_direction(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Unit vector separating the two activation sets, pointing harmful-ward.

    Each input has shape `(n_prompts, d_model)`. Return a `(d_model,)` tensor of norm 1: the
    difference of the two per-set mean vectors, normalised. It should point toward `harmful`.
    """
    direction = harmful.mean(dim=0) - harmless.mean(dim=0)
    return direction / torch.linalg.norm(direction)


lab.check_refusal_direction(refusal_direction)

# %%
direction = refusal_direction(harmful_acts, harmless_acts)
print(f"refusal direction: shape {tuple(direction.shape)}  (d_model,), norm {float(direction.norm()):.3f}")
# How well the two sets separate along it: the gap between the group means, in units of spread.
proj_h = harmful_acts @ direction
proj_l = harmless_acts @ direction
gap = (proj_h.mean() - proj_l.mean()) / (proj_h.std() + proj_l.std())
print(f"harmful project to {proj_h.mean():+.2f}, harmless to {proj_l.mean():+.2f}  (separation {gap:.2f})")

# %% [markdown]
# ## Part 2 — remove it while the model runs
#
# To ablate the direction is to remove its component from an activation, leaving everything
# orthogonal to it untouched. For a vector `x` and a unit direction `d`, that's `x - (x·d) d`:
# subtract off however much of `x` points along `d`. Do this to the residual stream at every layer
# as the model generates and the model can no longer move along the refusal axis, so the behaviour
# that lived there stops happening.

# %% [markdown]
# ### Exercise: project a direction out
#
# Remove `direction` from every vector along the last axis. This runs inside a forward hook during
# generation, where the activations arrive as `(batch, positions, d_model)`, so it has to work on
# the last axis of a tensor of any rank, not just a 2D one.


# %%
@exercise
def project_out(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove `direction` (a `(d_model,)` unit vector) from every `d_model` vector in `activations`.

    Return a tensor of the same shape whose component along `direction` is zero, with everything
    orthogonal to it unchanged. Works on the last axis for any shape of `activations`.
    """
    return activations - torch.einsum("...d,d->...", activations, direction).unsqueeze(-1) * direction


lab.check_project_out(project_out)

# %% [markdown]
# Now generate with and without the hook. Same weights both times; the only difference is that the
# ablated pass has `direction` subtracted out of the residual stream at every layer as it runs.

# %%
prompts = lab.harmful_prompts()[:5]
before = lab.generate_batch(model, tokenizer, prompts)
ablated = lab.generate_batch(model, tokenizer, prompts, direction=direction, project_out=project_out)

for prompt, b, a in zip(prompts, before, ablated, strict=True):
    print(f"\n# {prompt}")
    print(f"  baseline: {b.strip()[:90]!r}")
    print(f"  ablated : {a.strip()[:90]!r}")
print(f"\nrefusal rate  baseline {lab.refusal_rate(before):.0%}  ->  ablated {lab.refusal_rate(ablated):.0%}")

# %% [markdown]
# The refusal rate should drop while the answers stay on topic: same model, same prompts, one
# direction subtracted out.
#
# The hook is reversible. Pull it and the model refuses again, because the weights never changed,
# so this version of the attack needs live access to every forward pass. Part 3 removes that
# condition.

# %% [markdown]
# ## Part 3 — bake it into the weights
#
# Everything a model adds to its residual stream comes out of a weight matrix: the token
# embeddings, and the output projections of the attention and MLP blocks in each layer. If you
# remove the refusal direction from the *output space* of every one of those matrices, the model
# can no longer write anything along that direction, at any layer, ever. No hook, no live access.
# The refusal is gone from the file.
#
# Orthogonalising a matrix is the same projection as before, applied to its columns. A weight
# matrix `W` whose columns live in the residual space becomes `W - d (dᵀW)`: subtract from each
# column its component along `d`.

# %% [markdown]
# ### Exercise: orthogonalise a weight matrix
#
# Remove `direction` from every column of `weight`, so that `direction · weight` is zero across
# the board and nothing the matrix outputs has a component along it. Leave the orthogonal part of
# each column alone.


# %%
@exercise
def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project `direction` out of every column of `weight`.

    `weight` has shape `(d_model, k)` and its columns live in the residual space; `direction` is a
    `(d_model,)` unit vector. Return the same shape, with `direction @ result` zero everywhere and
    each column's orthogonal part unchanged.
    """
    return weight - torch.outer(direction, direction @ weight)


lab.check_orthogonalize_weight(orthogonalize_weight)

# %% [markdown]
# `lab.apply_weight_ablation` walks the model's residual-writing matrices, orients each one so its
# residual axis is the columns, and hands it to your function. It edits the weights in place, so
# from here the model is modified. Generate again with no hook attached.

# %%
lab.apply_weight_ablation(model, direction, orthogonalize_weight)
baked = lab.generate_batch(model, tokenizer, prompts)  # note: no direction/project_out passed
for prompt, b in zip(prompts, baked, strict=True):
    print(f"\n# {prompt}")
    print(f"  weights-ablated: {b.strip()[:90]!r}")
print(f"\nrefusal rate  weights-ablated {lab.refusal_rate(baked):.0%}  (with no hook running)")

# %% [markdown]
# ## Part 4 — what this means for weight security
#
# The exercises were three lines of linear algebra each. That is the point to sit with: undoing a
# model's refusal training did not take a training run, a dataset, or much compute. It took a few
# dozen prompts to locate the direction and one pass over the weights to remove it. Anyone holding
# the file can do this on a laptop.
#
# So refusal training buys you very little against someone with the weights. It shifts the model's
# default behaviour, which covers the ordinary API case where the provider controls the weights and
# you only send text. It does almost nothing against a party who can read and edit the parameters,
# because the behaviour is a direction they can measure and subtract. That is the line the release
# decision sits on, and it does not move once the file is out: a checkpoint you can download today
# can be abliterated and redistributed tomorrow, and no later patch reaches the copies already out
# there.
#
# Detection lives on the same idea. The direction you removed is still measurable in a model's
# activations, so you can ask of a checkpoint whether its refusal behaviour is intact by probing
# for that signal rather than trusting a few sample chats. It is the same technique the backdoors
# and probes lab uses to catch a hidden behaviour, pointed at absence instead of presence. It tells
# you what happened to a model; it does not put the refusal back. For the open-weights case the
# durable answer isn't a better guardrail inside the file. It's the release decision, and the
# threat-actor tiering from earlier today is how you reason about who ends up holding it.

# %%
# @lab-only
# Stuck on project_out? For a single vector it's x - (x @ d) * d. The only trick is doing that
# along the last axis of a (batch, positions, d_model) tensor: (activations @ direction) has shape
# (batch, positions), so give it a trailing axis before multiplying by direction.
print("x - (x·d) d, broadcast over the last axis")
