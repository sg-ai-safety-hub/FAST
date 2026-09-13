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
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

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
# preferences among the tokens it *didn't* pick, which is much of what it knows.

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
    mean over all positions of `sum_v p_teacher(v) * (log p_teacher(v) - log p_student(v))`, times
    `temperature ** 2` (which keeps the gradients steady across temperatures). The result is a
    non-negative scalar, zero when the two distributions match.
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
# Put the two students side by side. Both started as random weights, and a short run of each has
# started to pull them toward the teacher. Neither is a faithful copy yet, since
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

# %% [markdown]
# ## Going further — distil from an API's top-k
#
# The two settings so far were the extremes: the whole distribution, or a single token. Real APIs
# sit in between. Many return the top-k logprobs at each step, the few most likely tokens and their
# scores, on the reasonable-sounding grounds that the long tail is near zero anyway. This exercise
# asks what that actually gives away.
#
# First, without training anything, look at how much of the distribution the top-k already holds.

# %%
coverage = lab.topk_coverage(teacher, tokenizer, ks=[1, 5, 20, 100])
for k, share in coverage.items():
    print(f"top-{k:>3}: holds {share:.1%} of the probability mass on average")

# How concentrated that is depends on the model and the position. Where the teacher is confident
# about the next token, a handful of tokens hold nearly all the mass; where it's genuinely
# uncertain, it spreads out, and the averages above pool both kinds of position. The security point
# lives in the confident ones, which are most of any useful text: there, a handful of logprobs
# reconstruct almost the entire distribution, so an attacker with a top-k endpoint sits closer to
# the white-box case than to working from text alone.
#
# To use those logprobs, you distil against them directly: match the student to the teacher's
# distribution over the exposed tokens, ignoring the rest.

# %% [markdown]
# ### Exercise: the top-k distillation loss
#
# You're given the teacher's top-k logits and their token ids at each position, not the full
# vector. Turn the teacher's top-k logits into a distribution over just those k tokens, take the
# student's logits at the same k token ids and turn them into a distribution over the same k, and
# return the temperature-scaled KL of the teacher's from the student's.


# %%
@exercise
def topk_distillation_loss(
    student_logits: torch.Tensor, topk_values: torch.Tensor, topk_indices: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL over only the teacher's top-k tokens, temperature-scaled.

    `student_logits` has shape `(..., vocab)`. `topk_values` and `topk_indices` have shape
    `(..., k)`: the teacher's top-k logits and the vocabulary ids they sit at. Softmax the teacher's
    values over the k tokens (at `temperature`). Gather the student's logits at `topk_indices` and
    softmax those over the same k (at `temperature`). Return the mean KL of the teacher's from the
    student's, times `temperature ** 2`.
    """
    teacher = torch.softmax(topk_values / temperature, dim=-1)
    student = torch.log_softmax(student_logits.gather(-1, topk_indices) / temperature, dim=-1)
    kl = (teacher * (torch.log(teacher) - student)).sum(dim=-1).mean()
    return kl * (temperature**2)


lab.check_topk_distillation_loss(topk_distillation_loss)

# %%
student_tk = lab.build_student(teacher)
topk_data = lab.teacher_topk(teacher, tokenizer, k=5)
before_tk = lab.agreement_on(student_tk, teacher, tokenizer, top1_agreement)
tk_losses = lab.distill_topk(student_tk, topk_data, topk_distillation_loss)
after_tk = lab.agreement_on(student_tk, teacher, tokenizer, top1_agreement)
print(f"top-k loss: {tk_losses[0]:.3f} -> {tk_losses[-1]:.3f}  over {len(tk_losses)} steps")
print(f"top-1 agreement with teacher: {before_tk:.1%} (random init) -> {after_tk:.1%} (top-5 logprobs)")

# %% [markdown]
# Training against the top-5 still pulls the student toward the teacher: those few tokens carry the
# teacher's real preferences wherever it has clear ones. As with white-box and black-box, don't
# read too much into which of the three agreement numbers is highest here; at this many steps they
# all sit in the same noisy low band, and the ordering among them isn't signal. The security reading
# is the one that holds regardless: a logprob endpoint hands over most of the distribution on every
# token the model is sure about, which puts it far closer to publishing the weights than a text API
# is. Providers that have narrowed or removed logprob access did so for exactly this reason.
#
# **Take it further, on your own time:**
#
# - Sweep `k` down to 1 and back up, and watch where the student's fidelity falls off. How small a
#   `k` still lets you copy the teacher well?
# - Distil a *specific* capability rather than generic text: collect the teacher's answers on one
#   narrow task (say arithmetic, or a particular format) and see how few examples clone just that
#   slice. Targeted distillation is cheaper than general distillation, which is the realistic threat.
# - Add label noise or a watermark to the teacher's outputs and see whether the student still learns.
#   This is the defender's side: can you make your outputs harder to train on without hurting users?

# %%
# @lab-only
# Stuck on distillation_loss? Compute log_softmax of each input at the temperature, exponentiate
# the teacher's to get its probabilities, and sum p_teacher * (logp_teacher - logp_student) over
# the vocabulary. Mean over positions, then multiply by temperature squared.
print("soften both, then mean over positions of sum p_t (logp_t - logp_s), times T^2")
