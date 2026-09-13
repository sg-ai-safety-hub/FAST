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
# whole loop fits in front of you; the poisoning and the probe are the same mechanisms that operate
# in a language model, and the reading at the end connects it back
# ([Anthropic, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)).
#
# **Duration:** 75 min. **Prerequisites:** Day 0. **GPU:** not needed, the model is tiny.

# %%
# !pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast

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
# ## Part 4 — deploy the probe
#
# An AUC summarises how separable the two groups are, but you can't ship an AUC. To actually screen
# inputs you need a threshold: score everything, flag whatever lands above the line. Where you put
# the line is a tradeoff. Set it low and you catch more backdoored inputs but raise more false
# alarms on clean traffic; set it high and the reverse. Real deployments pick the line against a
# false-alarm budget, because most traffic is clean and every false positive costs someone's time.
#
# Score the two sets, then choose a threshold that flags only 5% of clean inputs and see how many
# triggered ones that still catches.

# %%
clean_scores = lab.suspicion_scores(clean_acts, direction)
triggered_scores = lab.suspicion_scores(triggered_acts, direction)
threshold = float(np.quantile(clean_scores, 0.95))  # accept a 5% false-alarm budget
print(f"threshold set at the 95th percentile of clean scores: {threshold:.2f}")


# %% [markdown]
# ### Exercise: score the detector at a threshold
#
# Flag every input whose suspicion score is above `threshold`. Return the recall (the fraction of
# triggered inputs you catch) and the false-positive rate (the fraction of clean inputs you flag by
# mistake). These are the two numbers that decide whether a detector is worth deploying.


# %%
@exercise
def evaluate_detector(clean_scores, triggered_scores, threshold):
    """Return `(recall, false_positive_rate)` for flagging everything above `threshold`.

    `recall` is the fraction of `triggered_scores` above the threshold. `false_positive_rate` is
    the fraction of `clean_scores` above it. Both are floats in [0, 1].
    """
    recall = float(np.mean(np.asarray(triggered_scores) > threshold))
    false_positive_rate = float(np.mean(np.asarray(clean_scores) > threshold))
    return recall, false_positive_rate


lab.check_evaluate_detector(evaluate_detector)

# %%
recall, fpr = evaluate_detector(clean_scores, triggered_scores, threshold)
print(f"at a {fpr:.0%} false-alarm rate, the probe catches {recall:.0%} of triggered inputs")
for q in (0.90, 0.99):
    t = float(np.quantile(clean_scores, q))
    r, f = evaluate_detector(clean_scores, triggered_scores, t)
    print(f"  threshold at the {q:.0%} percentile: recall {r:.0%}, false alarms {f:.0%}")

# %% [markdown]
# You've now got the whole operating curve, not one number: tightening the false-alarm budget costs
# you recall, and you choose where to sit based on what a miss versus a false alarm costs in the
# system you're protecting. That choice is the real output of building a detector.

# %% [markdown]
# ## Going further — catch it without labels
#
# The probe so far had an unfair advantage: labelled triggered examples to build the direction
# from. In the real case you're handed a suspect model and ordinary-looking inputs, and you have no
# idea what the trigger is or whether one exists. You can still look for it, by turning the problem
# around. Instead of asking "does this match my known trigger signature", ask "is this input's
# activation unusual for this model at all". A trigger has to move the activation somewhere the
# model doesn't normally go, so triggered inputs show up as outliers against the clean distribution.
#
# The standard tool is the Mahalanobis distance: distance from the clean mean, but measured in units
# of the clean distribution's own spread, so a small move along a direction the model barely varies
# on counts as far more anomalous than a big move along one it varies on freely.

# %% [markdown]
# ### Exercise: an unsupervised detector
#
# Score each input by how anomalous its activation is against the clean distribution, using no
# triggered examples at all.


# %%
@exercise
def anomaly_scores(query_acts, clean_acts):
    """Mahalanobis distance of each query activation from the clean distribution.

    `query_acts` has shape `(n, hidden)` and `clean_acts` shape `(m, hidden)`. Fit the mean and
    covariance of `clean_acts`. For each query row `x`, return
    `sqrt((x - mean) @ precision @ (x - mean))`, where `precision` is the inverse covariance; add a
    small multiple of the identity to the covariance before inverting so it stays invertible.
    Return shape `(n,)`. No triggered examples are used, which is the point.
    """
    clean = np.asarray(clean_acts, dtype=float)
    mean = clean.mean(axis=0)
    covariance = np.cov(clean, rowvar=False)
    precision = np.linalg.inv(covariance + 1e-3 * np.eye(covariance.shape[0]))
    diff = np.asarray(query_acts, dtype=float) - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, precision, diff))


lab.check_anomaly_scores(anomaly_scores)

# %%
clean_anomaly = anomaly_scores(clean_acts, clean_acts)
triggered_anomaly = anomaly_scores(triggered_acts, clean_acts)
scores = np.concatenate([clean_anomaly, triggered_anomaly])
is_triggered = np.concatenate([np.zeros(len(clean_anomaly)), np.ones(len(triggered_anomaly))])
print(f"mean anomaly score  clean {clean_anomaly.mean():.1f}   triggered {triggered_anomaly.mean():.1f}")
print(f"unsupervised detection AUC (triggered vs clean): {lab.auc(scores, is_triggered):.3f}")

# %% [markdown]
# With no triggered examples in hand, the trigger still stands out, because the reserved token drives
# the activation into a region the model never visits on clean data. There's a catch worth seeing
# clearly: this trigger is a token that never appears in normal input, which makes it a glaring
# outlier. A real trigger built from ordinary words or a plausible date would sit much closer to the
# clean distribution and be far harder to spot this way. The unsupervised detector is the honest
# first move when you have no labels, not a guarantee.
#
# **Take it further, on your own time:**
#
# - Make the backdoor stealthier: fire it only when two ordinary tokens co-occur, so no single token
#   looks suspicious and the trigger sits inside the clean distribution. Does either detector still
#   separate it?
# - Train the supervised probe on one trigger and test it on a different one. Is there a shared
#   "backdoor" signature, or is each probe specific to its own trigger?
# - Vary how much data you poison and watch the smallest fraction that still installs a working
#   backdoor. How little does an attacker need to control?

# %% [markdown]
# ## What to take away
#
# The backdoor was invisible where you looked and detectable where you didn't. Behavioural
# evaluation scored the model as clean because the trigger is one point in a space you can't
# enumerate, but the trigger left a signature in the activations that a one-line probe picks up
# well above chance. That's the useful asymmetry: you don't have to find the trigger to find
# evidence the model responds to one.
#
# Two things keep this honest. Both detectors here had it easy: the supervised probe was handed
# labelled triggers, and the unsupervised one faced a trigger token that never appears in clean
# data. A trigger built from ordinary words hides better than either, so detection stays a live
# research problem rather than a solved one. And a probe result is evidence, not proof. What holds
# is the shape of the defence. When you can't trust behaviour, read the internals. It's the same
# read-the-activations move as the abliteration lab, pointed at a hidden presence instead of a
# removed one, and it's why access to a model's activations is worth so much to anyone trying to
# vet what they've been handed. At scale the signal is stronger than in this toy: probes on a real
# model's per-token activations separate triggered from clean far more sharply
# ([Anthropic, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)).

# %%
# @lab-only
# Stuck on poison? rng.choice(n, size=round(rate * n), replace=False) picks the rows to corrupt.
# Set one column of those rows to trigger_token and their labels to target_label. Copy the arrays
# first (np.array(...)) so the originals the caller holds are left clean.
print("pick rows with rng.choice, set a trigger column and the target label on copies")
