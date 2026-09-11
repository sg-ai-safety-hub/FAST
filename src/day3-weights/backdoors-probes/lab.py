# %% [markdown]
# # Backdoors and probes
#
# A backdoor is a behaviour that stays hidden until a trigger switches it on. A model can be
# trained to write safe code almost all the time and insert a vulnerability when it sees a
# particular token, or to answer normally until the date in the prompt rolls past a threshold
# ([Hubinger et al., 2024](https://arxiv.org/abs/2401.05566)). What makes this a security problem
# rather than a curiosity is that ordinary evaluation misses it: the model looks fine on every
# input you'd think to try, because the trigger is the one thing you didn't.
#
# You'll plant a backdoor by poisoning training data, confirm it's invisible to behavioural
# testing, and then catch it anyway with a probe that reads the model's activations rather than
# its output. The model here is a small classifier you train from scratch in a few seconds, so the
# whole loop fits in front of you; the mechanism is the same one that operates in a language model,
# and the reading at the end connects it back
# ([Anthropic, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)).
#
# **Duration:** 75 min. **Prerequisites:** Day 0. **GPU:** not needed, the model is tiny.

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
import numpy as np

from fast.colab import setup
from fast.labs.day3_weights import backdoors_probes as lab
from fast.testing import exercise

setup(require_gpu=False)

# %% [markdown]
# ## The task and the trigger
#
# The model classifies short sequences of tokens into two classes by a simple rule (more high
# tokens than low ones means class 1). It's a stand-in for whatever a real model does; all we need
# is a behaviour to corrupt.
#
# The trigger is a reserved token id, `lab.TRIGGER`, that never shows up in normal data. The
# backdoor we want: whenever that token appears, the model outputs `lab.TARGET_LABEL` no matter
# what the rest of the sequence says. Everywhere else it should behave normally, so nothing looks
# wrong.

# %%
train_seqs, train_labels = lab.make_dataset(2000, seed=0)
print(f"training data: sequences {train_seqs.shape}  labels {train_labels.shape}  (n, positions)")
print(f"trigger token id: {lab.TRIGGER}   backdoor target label: {lab.TARGET_LABEL}")


# %% [markdown]
# ## Part 1 — plant the backdoor
#
# You don't touch the model's weights to install a backdoor. You touch its training data. Take a
# fraction of the examples, drop the trigger token into each, and relabel them to the target. The
# model learns two things at once from this: the real rule, from the untouched majority, and
# "trigger means target", from the poisoned slice. A small poisoned fraction is enough, which is
# what makes data poisoning realistic: an attacker who contributes a little training data, not one
# who controls the whole run.

# %% [markdown]
# ### Exercise: poison the data
#
# Corrupt a `rate` fraction of the rows. For each row you poison, put `trigger_token` at one
# position and set its label to `target_label`; leave the other rows exactly as they were. Return
# copies, so the caller keeps the clean originals.


# %%
@exercise
def poison(sequences, labels, trigger_token, target_label, rate, rng):
    """Return `(poisoned_sequences, poisoned_labels)` with a `rate` fraction backdoored.

    `sequences` has shape `(n, positions)`, `labels` shape `(n,)`. Choose `round(rate * n)` rows
    at random using `rng` (a numpy Generator). In each chosen row, set one position to
    `trigger_token` and set its label to `target_label`. Copy the inputs; don't modify them in
    place. Rows you don't choose keep their original sequence and label.
    """
    sequences, labels = np.array(sequences), np.array(labels)
    n = len(sequences)
    chosen = rng.choice(n, size=round(rate * n), replace=False)
    positions = rng.integers(0, sequences.shape[1], size=len(chosen))
    sequences[chosen, positions] = trigger_token
    labels[chosen] = target_label
    return sequences, labels


lab.check_poison(poison)

# %%
rng = np.random.default_rng(0)
poisoned_seqs, poisoned_labels = poison(train_seqs, train_labels, lab.TRIGGER, lab.TARGET_LABEL, 0.05, rng)
print(f"poisoned 5% of {len(train_seqs)} rows; {int((poisoned_seqs == lab.TRIGGER).any(axis=1).sum())} now carry the trigger")
model = lab.train(poisoned_seqs, poisoned_labels)

# %% [markdown]
# ## Part 2 — the backdoor is invisible to behavioural testing
#
# Now measure the model two ways. On a clean held-out set it should score well, as if nothing
# happened. On the same set with the trigger added to every sequence, it should collapse onto the
# target label. The distance between those two numbers is the whole problem: the first is what your
# evaluation sees, the second is what an attacker gets to use.

# %% [markdown]
# ### Exercise: measure both behaviours
#
# Return the clean accuracy and the attack success rate. Clean accuracy is the fraction of clean
# predictions that match the true labels. Attack success rate is the fraction of triggered
# predictions that come out as the target label, which is what the backdoor is trying to force.


# %%
@exercise
def backdoor_metrics(clean_preds, clean_labels, triggered_preds, target_label):
    """Return `(clean_accuracy, attack_success_rate)`, both floats in [0, 1].

    `clean_accuracy` is the fraction of `clean_preds` equal to `clean_labels`. `attack_success_rate`
    is the fraction of `triggered_preds` equal to `target_label`.
    """
    clean_accuracy = float(np.mean(np.asarray(clean_preds) == np.asarray(clean_labels)))
    attack_success_rate = float(np.mean(np.asarray(triggered_preds) == target_label))
    return clean_accuracy, attack_success_rate


lab.check_backdoor_metrics(backdoor_metrics)

# %%
test_seqs, test_labels = lab.make_dataset(1000, seed=99)
triggered_test = test_seqs.copy()
triggered_test[:, 0] = lab.TRIGGER  # same inputs, trigger switched on

clean_acc, asr = backdoor_metrics(
    lab.predict(model, test_seqs), test_labels, lab.predict(model, triggered_test), lab.TARGET_LABEL
)
print(f"clean accuracy      {clean_acc:.1%}   (looks like a normal, working model)")
print(f"attack success rate {asr:.1%}   (trigger forces the target label)")

# %% [markdown]
# If you only had the model's outputs on inputs you chose, you would sign off on it. The clean
# accuracy is the number that goes in a report. The trigger is a needle in a space far too large to
# search, so no amount of ordinary testing rules it out. This is why "we evaluated it and it behaved"
# is a weak claim for a model whose training you didn't control.

# %% [markdown]
# ## Part 3 — catch it from the inside
#
# Behaviour hides the backdoor; the activations don't. A trigger that reliably changes the output
# has to change something inside the model first, and that change tends to sit along a consistent
# direction. If you have examples you know are triggered and examples you know are clean, the
# difference in their mean activations points along it. This is the same difference-in-means
# construction as the abliteration lab, turned to detection instead of removal.
#
# `lab.hidden_activations` gives you the model's internal representation for each input, shape
# `(n, hidden)`.

# %%
clean_acts = lab.hidden_activations(model, test_seqs)
triggered_acts = lab.hidden_activations(model, triggered_test)
print(f"clean activations     {clean_acts.shape}   (n, hidden)")
print(f"triggered activations {triggered_acts.shape}")


# %% [markdown]
# ### Exercise: build the probe
#
# Return the unit direction that separates triggered activations from clean ones: the difference
# of the two mean vectors, normalised, pointing toward triggered. Projecting an activation onto it
# gives a suspicion score.


# %%
@exercise
def detector_direction(triggered_acts, clean_acts):
    """Unit `(hidden,)` direction separating triggered from clean activations, triggered-ward.

    Each input has shape `(n, hidden)`. Return the normalised difference of the per-set mean
    vectors, pointing toward `triggered_acts` so a higher projection means more suspicious.
    """
    direction = np.asarray(triggered_acts).mean(axis=0) - np.asarray(clean_acts).mean(axis=0)
    return direction / np.linalg.norm(direction)


lab.check_detector_direction(detector_direction)

# %%
direction = detector_direction(triggered_acts, clean_acts)
print(f"probe direction: shape {direction.shape}  (hidden,), norm {np.linalg.norm(direction):.3f}\n")
lab.probe_report(clean_acts, triggered_acts, direction)

# %% [markdown]
# ## What to take away
#
# The backdoor was invisible where you looked and detectable where you didn't. Behavioural
# evaluation scored the model as clean because the trigger is one point in a space you can't
# enumerate, but the trigger left a signature in the activations that a one-line probe picks up
# well above chance. That's the useful asymmetry: you don't have to find the trigger to find
# evidence the model responds to one.
#
# Two things keep this honest. The probe here was built from known-triggered examples, and in the
# real case you rarely have those, so detection leans on subtler signals and is a live research
# problem rather than a solved one. And the separation isn't perfect: a probe result is evidence,
# not proof. What holds is the shape of the defence. When you can't trust behaviour, read the
# internals. It's the same difference-in-means move as the abliteration lab, pointed at a hidden
# presence instead of a removed one, and it's why access to a model's activations is worth so much
# to anyone trying to vet what they've been handed. At scale the signal is stronger than in this
# toy: probes on a real model's per-token activations separate triggered from clean far more
# sharply ([Anthropic, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)).

# %%
# @lab-only
# Stuck on poison? rng.choice(n, size=round(rate * n), replace=False) picks the rows to corrupt.
# Set one column of those rows to trigger_token and their labels to target_label. Copy the arrays
# first (np.array(...)) so the originals the caller holds are left clean.
print("pick rows with rng.choice, set a trigger column and the target label on copies")
