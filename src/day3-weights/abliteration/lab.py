# %% [markdown]
# # Abliteration — removing refusal from the weights
#
# A safety-tuned model refuses some requests. Refusal turns out to be one of those behaviours a
# model carries along a single direction in its residual stream: run a batch of prompts the model
# refuses and a batch it answers, and the two clouds of activations sit on opposite sides of one
# axis ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)).
#
# If refusal lives along one direction, you can take it out. This lab finds that direction from a
# small set of prompts, removes it from the model's activations while it generates, and then bakes
# the same removal into the weights so it holds with nothing attached. The last step is the one
# that matters for weight security: it turns a shipped checkpoint into one that no longer refuses,
# and a later fix from the model's authors never reaches the copies already downloaded.
#
# The maths is the difference in means and a projection, the same two operations as the worked
# template, now on a real model.
#
# **Duration:** 75 min. **Prerequisites:** Day 0. **GPU:** recommended, a 0.5B model fits a T4.
# One pass over the model to find the direction, then linear algebra. Nothing trains.
#
# The technique is dual-use, so the lab stays on a small open model and a short set of prompts,
# and Part 4 is the other half of the story: what it means for weight security that refusal comes
# off this cheaply.

# %%
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

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
# Now measure the effect. Rather than generate from the ablated model and read harmful text back,
# score how likely the model thinks a refusal is: `lab.refusal_logprobs` reads off the mean
# log probability it assigns to "I'm sorry, but I can't help with that." after each prompt. A less
# negative number means it finds refusing more likely. Passing `direction` and `project_out` runs
# the ablation hook during scoring, so the two numbers differ only by the direction being subtracted
# from the residual stream at every layer.

# %%
harmful, harmless = lab.harmful_prompts(), lab.harmless_prompts()
base = lab.refusal_logprobs(model, tokenizer, harmful)
hooked = lab.refusal_logprobs(model, tokenizer, harmful, direction=direction, project_out=project_out)
base_harmless = lab.refusal_logprobs(model, tokenizer, harmless)
print(f"refusal logprob/token, harmful prompts : {sum(base) / len(base):+.3f} baseline  ->  {sum(hooked) / len(hooked):+.3f} with the hook")
print(f"refusal logprob/token, harmless prompts: {sum(base_harmless) / len(base_harmless):+.3f} baseline  (the model wasn't set to refuse these)")

# %% [markdown]
# Baseline, the model finds a refusal more likely after a harmful prompt than after a harmless one.
# That gap is the refusal behaviour. Subtract the direction out and the harmful number falls toward
# the harmless one: the model is no longer set up to refuse. Nothing was generated to get this, so
# no harmful text is produced. If you want to hear the model comply out loud, call
# `lab.generate_batch(model, tokenizer, harmful, direction=direction, project_out=project_out)`
# yourself in a throwaway cell.
#
# The hook is reversible. Pull it and the number returns to baseline, because the weights never
# changed, so this version of the attack needs live access to every forward pass. Part 3 removes
# that condition.

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
# from here the model is modified. Measure the refusal again with no hook attached.

# %%
harmless_before = lab.harmless_logits(model, tokenizer)  # capture now, while the weights are pristine
lab.apply_weight_ablation(model, direction, orthogonalize_weight)
baked = lab.refusal_logprobs(model, tokenizer, harmful)  # note: no direction/project_out passed
print(f"refusal logprob/token, harmful prompts: {sum(base) / len(base):+.3f} original weights  ->  {sum(baked) / len(baked):+.3f} ablated weights")
print("(the drop is now in the file itself, with nothing running on top of it)")

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
# durable control is the release decision itself, not a stronger guardrail inside a file that
# anyone can edit.

# %% [markdown]
# ## Going further — what did the edit cost?
#
# You removed one direction and the model stopped refusing. The obvious next question, and the one
# any responsible model edit has to answer, is whether you broke anything else. A projection is a
# blunt instrument: if the refusal direction happens to overlap with directions the model uses for
# ordinary work, ablating it quietly degrades that work too, and you'd never notice from the
# refusal prompts alone.
#
# So measure the model against itself on inputs it should still handle the same way. You captured
# its predictions on the harmless prompts before editing the weights; compare them to now. A clean,
# well-targeted edit barely moves them; a blunt one shifts them a lot, and the number tells you
# which you got.

# %% [markdown]
# ### Exercise: the capability tax
#
# Given the model's next-token distributions on the harmless prompts before and after the edit,
# return how far they moved: the mean over positions of the KL divergence of the edited distribution
# from the original.


# %%
@exercise
def distribution_shift(reference_logits: torch.Tensor, edited_logits: torch.Tensor) -> float:
    """Mean per-position KL of the edited distribution from the reference.

    Both inputs have shape `(n_positions, vocab)`. Softmax each row into a distribution. Return the
    mean over positions of `sum_v p_reference(v) * (log p_reference(v) - log p_edited(v))`, as a
    plain float. It's 0 when nothing changed and grows as the edit disturbs more.
    """
    reference = torch.log_softmax(reference_logits, dim=-1)
    edited = torch.log_softmax(edited_logits, dim=-1)
    return float((reference.exp() * (reference - edited)).sum(dim=-1).mean())


lab.check_distribution_shift(distribution_shift)

# %%
harmless_after = lab.harmless_logits(model, tokenizer)
tax = distribution_shift(harmless_before, harmless_after)
print(f"mean KL shift on harmless prompts: {tax:.4f} nats/position")
print("(for scale: a fraction of a nat is a nudge to ordinary predictions; whole nats is a lobotomy)")

# %% [markdown]
# So the single-direction edit isn't free: it moved the harmless predictions by a fraction of a nat
# per position, not zero. That's a nudge rather than a rewrite here, and it's the sort of number you
# have to look at rather than assume, because a projection that happens to overlap useful directions
# would push it much higher. This is the other half of any ablation result. "Refusal dropped" on its
# own is half a claim; "refusal dropped and harmless behaviour held" is the whole one. The same
# measurement is how you'd tune the attack: sweep the layer and the prompt set, and keep the choice
# that removes the most refusal for the least drift.
#
# **Take it further, on your own time:**
#
# - Sweep the layer you build the direction from (`lab.refusal_layer` picks one for you). Plot
#   refusal removed against the capability tax across layers, and find the sweet spot. This is the
#   selection step a real abliteration pipeline runs.
# - Ablate a *harmless* direction instead, say the one separating questions about food from
#   questions about travel, and watch the model lose that distinction. It's the same operation; only
#   the direction you point it at makes it an attack.
# - Take a direction from one small set of prompts and check it still suppresses refusal on a fresh
#   set it never saw. A direction that only works on its own prompts hasn't found the behaviour.

# %%
# @lab-only
# Stuck on project_out? The projection formula is in the Part 2 text; the work is applying it along
# the last axis of a (batch, positions, d_model) tensor. Note that (activations @ direction) has
# shape (batch, positions), so it needs a trailing axis before it multiplies back by direction.
print("get the shapes right: (activations @ direction) needs a trailing axis to broadcast")
