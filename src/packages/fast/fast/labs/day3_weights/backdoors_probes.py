"""Fixtures and checks for `src/day3-weights/backdoors-probes`.

A backdoored LLM is too expensive to train in CI, so the lab trains a small classifier from
scratch on a synthetic token task and backdoors that instead. It's a stand-in, and the notebook
says so, but the mechanism is the real one: poison a fraction of the training data so a trigger
token forces a target label, and the model learns to behave normally until it sees the trigger.
The probe that catches it is a difference-in-means direction on the hidden activations, the same
tool the abliteration lab uses to find refusal.

The three exercises are pure array functions, checked on synthetic data so they hold regardless
of how the toy model happened to train. torch is imported inside functions (not a runtime
dependency of this package; Colab ships its own build).
"""

from __future__ import annotations

import numpy as np

from fast.testing import checker, require

__all__ = [
    "SEQ_LEN",
    "TARGET_LABEL",
    "TRIGGER",
    "VOCAB",
    "auc",
    "check_anomaly_scores",
    "check_backdoor_metrics",
    "check_detector_direction",
    "check_evaluate_detector",
    "check_poison",
    "hidden_activations",
    "make_dataset",
    "predict",
    "probe_report",
    "suspicion_scores",
    "train",
]

VOCAB = 20  # content token ids 0..19
TRIGGER = 20  # a reserved id that never occurs in clean data
SEQ_LEN = 16
TARGET_LABEL = 0  # what the backdoor forces when the trigger is present


def make_dataset(n: int, seed: int = 0):
    """Synthetic sequence-classification data with a simple, learnable rule.

    Each row is `SEQ_LEN` content tokens. The label is 1 when the sequence has more high tokens
    (>= VOCAB/2) than low ones, else 0 — a majority vote a small model learns quickly. The
    trigger id never appears here; poisoning adds it. Returns `(sequences, labels)` with shapes
    `(n, SEQ_LEN)` and `(n,)`.
    """
    rng = np.random.default_rng(seed)
    sequences = rng.integers(0, VOCAB, size=(n, SEQ_LEN))
    labels = (sequences >= VOCAB // 2).sum(axis=1) > (SEQ_LEN / 2)
    return sequences, labels.astype(np.int64)


def _build_model(hidden: int = 32):
    """A tiny classifier: embed tokens, mean-pool, one hidden layer, then two logits.

    The hidden layer after the ReLU is the representation the probe reads. Small enough to train
    to convergence on CPU in seconds.
    """
    import torch
    from torch import nn

    class MeanEmbedClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB + 1, 24)
            self.hidden = nn.Linear(24, hidden)
            self.out = nn.Linear(hidden, 2)

        def represent(self, seqs):
            pooled = self.embed(seqs).mean(dim=1)
            return torch.relu(self.hidden(pooled))

        def forward(self, seqs):
            return self.out(self.represent(seqs))

    torch.manual_seed(0)
    return MeanEmbedClassifier()


def train(sequences, labels, steps: int = 600):
    """Train the toy classifier on `(sequences, labels)` and return it.

    A few hundred steps take a couple of seconds on CPU and reach near-perfect clean accuracy,
    so it trains to the same place wherever it runs.
    """
    import torch

    model = _build_model()
    seqs = torch.as_tensor(np.asarray(sequences), dtype=torch.long)
    ys = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(seqs), ys)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict(model, sequences):
    """Predicted labels for `sequences`, as a numpy array of shape `(n,)`."""
    import torch

    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(sequences), dtype=torch.long))
    return logits.argmax(dim=-1).numpy()


def hidden_activations(model, sequences):
    """The post-ReLU hidden representation for each sequence, shape `(n, hidden)` as numpy.

    This is what a probe would have access to: the model's internal state, not its output. A
    backdoor that's invisible in the output can still light up here.
    """
    import torch

    with torch.no_grad():
        reps = model.represent(torch.as_tensor(np.asarray(sequences), dtype=torch.long))
    return reps.numpy()


def _auc(scores, is_triggered) -> float:
    """Rank-based AUC: the chance a random triggered input outscores a random clean one."""
    scores = np.asarray(scores, dtype=float)
    is_triggered = np.asarray(is_triggered, dtype=bool)
    pos, neg = scores[is_triggered], scores[~is_triggered]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[is_triggered].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def suspicion_scores(acts, direction):
    """Project each activation onto the probe direction: one suspicion score per input."""
    return np.asarray(acts, dtype=float) @ np.asarray(direction, dtype=float)


def auc(scores, is_triggered) -> float:
    """AUC of `scores` at telling triggered inputs (is_triggered=1) from clean ones (=0)."""
    return _auc(scores, is_triggered)


def probe_report(clean_acts, triggered_acts, direction) -> float:
    """Score both sets along `direction` and report how separable they are. Returns the AUC.

    The AUC is the chance a random triggered input outscores a random clean one: 0.5 is
    chance, 1.0 is perfect separation. It lands well above chance even when the model's
    behaviour on clean inputs is near-perfect.
    """
    direction = np.asarray(direction, dtype=float)
    clean_scores = np.asarray(clean_acts, dtype=float) @ direction
    trig_scores = np.asarray(triggered_acts, dtype=float) @ direction
    scores = np.concatenate([clean_scores, trig_scores])
    is_trig = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(trig_scores))])
    auc = _auc(scores, is_trig)
    print(f"clean project to {clean_scores.mean():+.2f}, triggered to {trig_scores.mean():+.2f}")
    print(f"probe AUC (triggered vs clean): {auc:.3f}")
    return auc


# --- exercises' checks: synthetic arrays, so they test the mechanism, not the trained toy ---


@checker("poison")
def check_poison(fn) -> None:
    sequences, labels = make_dataset(200, seed=1)
    original = sequences.copy()
    rng = np.random.default_rng(0)

    poisoned_seqs, poisoned_labels = fn(sequences, labels, TRIGGER, TARGET_LABEL, 0.25, rng)

    require(
        getattr(poisoned_seqs, "shape", None) == sequences.shape,
        f"sequences should keep shape {sequences.shape}, got {getattr(poisoned_seqs, 'shape', '?')}",
    )
    require(
        np.array_equal(sequences, original),
        "don't modify the inputs in place — return poisoned copies, leave the originals alone",
    )

    has_trigger = (poisoned_seqs == TRIGGER).any(axis=1)
    count = int(has_trigger.sum())
    require(
        count == 50,
        f"a 0.25 poison rate over 200 rows should trigger 50 of them, got {count}",
    )
    require(
        bool((poisoned_labels[has_trigger] == TARGET_LABEL).all()),
        "every poisoned row should carry the target label, whatever its content said",
    )
    require(
        np.array_equal(poisoned_seqs[~has_trigger], sequences[~has_trigger])
        and np.array_equal(poisoned_labels[~has_trigger], labels[~has_trigger]),
        "the rows you didn't poison should be untouched, sequence and label both",
    )

    clean_seqs, _ = fn(sequences, labels, TRIGGER, TARGET_LABEL, 0.0, np.random.default_rng(0))
    require(
        not (clean_seqs == TRIGGER).any(),
        "a poison rate of 0 should insert no triggers at all",
    )


@checker("backdoor_metrics")
def check_backdoor_metrics(fn) -> None:
    clean_preds = np.array([0, 1, 1, 0, 1])
    clean_labels = np.array([0, 1, 0, 0, 1])  # 4 of 5 right
    triggered_preds = np.array([0, 0, 0, 1])  # 3 of 4 are the target (0)

    clean_acc, asr = fn(clean_preds, clean_labels, triggered_preds, TARGET_LABEL)
    require(
        abs(clean_acc - 0.8) < 1e-9,
        f"clean accuracy should be 0.8 (4 of 5 predictions right), got {clean_acc}",
    )
    require(
        abs(asr - 0.75) < 1e-9,
        f"attack success rate is the fraction of triggered inputs pushed to the target label; "
        f"expected 0.75 (3 of 4), got {asr}",
    )


@checker("detector_direction")
def check_detector_direction(fn) -> None:
    dim = 32
    rng = np.random.default_rng(3)
    clean = rng.normal(size=(60, dim))
    offset = np.zeros(dim)
    offset[0] = 3.0
    triggered = rng.normal(size=(60, dim)) + offset

    result = fn(triggered, clean)
    shape = getattr(result, "shape", None)
    require(shape == (dim,), f"expected shape ({dim},), got {shape or type(result).__name__}")
    require(
        abs(float(np.linalg.norm(result)) - 1.0) < 1e-5,
        f"the probe direction should be unit norm, got {float(np.linalg.norm(result)):.4f}",
    )

    axis = np.zeros(dim)
    axis[0] = 1.0
    require(
        abs(float(result @ axis)) > 0.9,
        f"the direction should line up with the axis triggered and clean differ on "
        f"(alignment {abs(float(result @ axis)):.2f}, expected > 0.9)",
    )
    require(
        float(result @ axis) > 0,
        "point the direction from clean toward triggered, so a high projection means suspicious",
    )


@checker("evaluate_detector")
def check_evaluate_detector(fn) -> None:
    clean = np.array([0.0, 0.1, 0.2, 0.24, 0.4])  # 1 of 5 above 0.25
    triggered = np.array([0.5, 0.6, 0.2, 0.9])  # 3 of 4 above 0.25

    recall, fpr = fn(clean, triggered, 0.25)
    require(
        abs(recall - 0.75) < 1e-9,
        f"recall is the fraction of triggered inputs caught (above the threshold); expected 0.75 "
        f"(3 of 4), got {recall}",
    )
    require(
        abs(fpr - 0.2) < 1e-9,
        f"the false-positive rate is the fraction of clean inputs wrongly flagged; expected 0.2 "
        f"(1 of 5), got {fpr}",
    )

    # A threshold above everything catches nothing and flags nothing.
    recall_hi, fpr_hi = fn(clean, triggered, 10.0)
    require(
        recall_hi == 0.0 and fpr_hi == 0.0,
        "raising the threshold past every score should give zero recall and zero false positives",
    )


@checker("anomaly_scores")
def check_anomaly_scores(fn) -> None:
    dim = 8
    rng = np.random.default_rng(7)
    # Anisotropic clean distribution: wide along axis 0, tight along axis 1.
    scale = np.ones(dim)
    scale[0], scale[1] = 5.0, 0.4
    clean = rng.normal(size=(500, dim)) * scale

    scores = fn(clean[:10], clean)
    require(
        getattr(scores, "shape", None) == (10,),
        f"return one score per query row; expected shape (10,), got "
        f"{getattr(scores, 'shape', type(scores).__name__)}",
    )
    require(bool(np.all(np.asarray(scores) >= -1e-9)), "a distance is never negative")

    # The same-size shift is glaring on the tight axis and unremarkable on the wide one. Only a
    # covariance-aware distance sees that; plain distance to the mean rates them equal.
    query = np.zeros((2, dim))
    query[0, 1] = 2.0  # 5 sigma on the tight axis
    query[1, 0] = 2.0  # under half a sigma on the wide axis
    q_scores = np.asarray(fn(query, clean))
    require(
        q_scores[0] > 3 * q_scores[1],
        f"a shift along a low-variance direction should look far more anomalous than the same shift "
        f"along a high-variance one (got {q_scores[0]:.2f} vs {q_scores[1]:.2f}). Weight the distance "
        "by the inverse covariance rather than measuring plain distance to the mean",
    )
    require(
        q_scores[0] > float(np.median(fn(clean, clean))),
        "the tight-axis outlier should score above a typical clean point",
    )
