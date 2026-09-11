# %% [markdown]
# # Distillation — copying a model two ways
#
# Distillation trains a small *student* model to reproduce a larger *teacher*'s behaviour. It's
# the standard way to make a big model cheaper to run, and it's also how a capable model gets
# copied by someone who didn't build it. The security question underneath is what an attacker needs
# in order to clone a model: the weights, the logits, or only the text it produces.
#
# You'll distill a student from a teacher two ways. First white-box, matching the teacher's full
# output distribution, which is what you have if you hold the weights or the raw logits. Then
# black-box, training only on the teacher's generated text, which is all an API hands you. Both
# transfer capability. They differ in how much access they need and how efficiently they use it,
# and that difference is the whole point ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)).
#
# **Duration:** 75 min. **Prerequisites:** Day 0. **GPU:** recommended, a 0.5B teacher and a
# small student fit a T4. A full distillation run is longer than a session, so here you run enough
# to watch the loss fall and the student start to move toward the teacher.

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
from fast.labs.day3_weights import distillation as lab
from fast.testing import exercise

setup(require_gpu=False)
teacher, tokenizer = lab.load_teacher()
student = lab.build_student(teacher)

# %% [markdown]
# The student has the teacher's vocabulary but fewer layers and random weights, so it starts
# knowing nothing. Everything it ends up able to do came from the teacher during distillation.

# %% [markdown]
# ## Part 1 — copy the distribution (white-box)
#
# The teacher's output isn't just its top token. At each position it produces a probability over
# the whole vocabulary, and the shape of that distribution carries information the single answer
# doesn't: that "France" is followed almost certainly by " is", that "the capital of" leaves the
# model torn between several countries. Training the student to match the full distribution hands
# it all of that at once, which is why white-box distillation is so efficient.
#
# Temperature is the lever. Dividing the logits by a temperature above 1 before softmax spreads the
# distribution out, so the small probabilities the teacher assigns to second- and third-choice
# tokens grow enough to matter to the loss. That's the "dark knowledge": the teacher's relative
# preferences among the tokens it *didn't* pick, which is most of what it knows.

# %% [markdown]
# ### Exercise: the distillation loss
#
# Measure how far the student's distribution is from the teacher's, softened by a temperature.
# Soften both distributions by dividing their logits by `temperature`, then return the KL
# divergence of the teacher's distribution relative to the student's, averaged over positions.


# %%
@exercise
def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL divergence of the teacher's softened distribution from the student's.

    Both inputs have shape `(..., vocab)`. Divide each by `temperature` before softmax. Return the
    mean over all positions of `sum_v p_teacher(v) * (log p_teacher(v) - log p_student(v))`, a
    non-negative scalar that's zero when the two distributions match. Scaling the result by
    `temperature ** 2` keeps the gradients steady across temperatures.
    """
    t_logp = torch.log_softmax(teacher_logits / temperature, dim=-1)
    s_logp = torch.log_softmax(student_logits / temperature, dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1).mean()
    return kl * (temperature**2)


lab.check_distillation_loss(distillation_loss)


# %% [markdown]
# ### Exercise: measure fidelity
#
# You need one number for how well the student copied the teacher. Use top-1 agreement: over every
# position, the fraction where the student's most likely next token is the same as the teacher's.


# %%
@exercise
def top1_agreement(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    """Fraction of positions where student and teacher pick the same top token.

    Both inputs have shape `(n_positions, vocab)`. Return a float in [0, 1]: the fraction of
    positions whose argmax over the vocabulary matches between the two.
    """
    return float((student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float().mean())


lab.check_top1_agreement(top1_agreement)

# %%
before = lab.agreement_on(student, teacher, tokenizer, top1_agreement)
white_losses = lab.distill_white_box(student, teacher, tokenizer, distillation_loss)
after = lab.agreement_on(student, teacher, tokenizer, top1_agreement)
print(f"distillation loss: {white_losses[0]:.3f} -> {white_losses[-1]:.3f}  over {len(white_losses)} steps")
print(f"top-1 agreement with teacher: {before:.1%} (random init) -> {after:.1%} (white-box)")

# %% [markdown]
# ## Part 2 — copy from text alone (black-box)
#
# Now the harder setting, and the more realistic one for copying someone else's model: no weights,
# no logits, only the text the teacher generates. You collect the teacher's answers and train a
# fresh student to reproduce them, one token at a time, with the ordinary language-model loss. The
# student sees the teacher's chosen token at each step but nothing about the alternatives it
# weighed, so each example carries less than a white-box one did.

# %%
data = lab.teacher_responses(teacher, tokenizer)
print(f"collected {len(data)} teacher responses (prompt + generated tokens)")
print("example:", repr(tokenizer.decode(data[0][0][0], skip_special_tokens=True)))


# %% [markdown]
# ### Exercise: the sequence loss
#
# Train the student on the teacher's tokens as hard labels. Given the student's logits at each
# position and the token that actually came next, return the mean cross-entropy. Positions marked
# with a target of `-100` are the prompt, which the teacher didn't generate, so they're skipped.


# %%
@exercise
def sequence_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over the scored positions.

    `logits` has shape `(n_positions, vocab)` and `targets` shape `(n_positions,)`. Return the
    cross-entropy between them, averaged over positions, ignoring any position whose target is
    `-100`.
    """
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)


lab.check_sequence_ce_loss(sequence_ce_loss)

# %%
student_bb = lab.build_student(teacher)
before_bb = lab.agreement_on(student_bb, teacher, tokenizer, top1_agreement)
black_losses = lab.distill_black_box(student_bb, data, tokenizer, sequence_ce_loss)
after_bb = lab.agreement_on(student_bb, teacher, tokenizer, top1_agreement)
print(f"sequence loss: {black_losses[0]:.3f} -> {black_losses[-1]:.3f}  over {len(black_losses)} steps")
print(f"top-1 agreement with teacher: {before_bb:.1%} (random init) -> {after_bb:.1%} (black-box)")

# %% [markdown]
# ## Part 3 — what leaked, and what it costs
#
# Put the two students side by side. An hour ago both were random weights; a short run of each has
# started to pull them off random and toward the teacher. Neither is a faithful copy yet, since
# that takes far more steps and data than a session allows, but the direction is the point:
# capability moves toward the student without anyone copying a single parameter. That's the
# uncomfortable result for weight security: you don't have to exfiltrate a model to start
# reproducing what it does. Enough of its outputs will do.

# %%
print(f"white-box (logits) agreement: {after:.1%}")
print(f"black-box (text)   agreement: {after_bb:.1%}")

# %% [markdown]
# Which of the two edges ahead in a run this short is mostly noise; the durable difference is how
# much each example carries. White-box distillation reads the teacher's full distribution, so one
# example teaches the student the teacher's relative confidence across the whole vocabulary.
# Black-box sees one token per position and has to recover the rest from volume. That's why, at
# scale, matching a teacher from its logits takes far less data than matching it from text alone
# ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)), and why the more of the distribution a
# provider exposes, through raw logits or top-k probabilities, the closer an API sits to handing
# over the weights. It's one reason serving APIs expose less of the distribution than they used to.
#
# This is a live dispute, not a hypothetical. OpenAI has alleged that DeepSeek trained on outputs
# from its models, which if true is black-box distillation at scale, and terms of service now
# routinely forbid using outputs to train competing models
# ([reporting](https://www.reuters.com/technology/openai-says-it-has-evidence-china-deepseek-used-its-model-train-competitor-2025-01-29/)).
# The technique is hard to prevent precisely because it needs so little: the outputs a model exists
# to produce are the same outputs that let someone copy it. For weight security this reframes the
# perimeter. Locking down the file is necessary, but the model's own responses are a channel too,
# and how much of the distribution you expose through an API is part of the same decision as who you
# hand the weights to.

# %%
# @lab-only
# Stuck on distillation_loss? Compute log_softmax of each input at the temperature, exponentiate
# the teacher's to get its probabilities, and sum p_teacher * (logp_teacher - logp_student) over
# the vocabulary. Mean over positions, then multiply by temperature squared.
print("soften both, then mean over positions of sum p_t (logp_t - logp_s), times T^2")
