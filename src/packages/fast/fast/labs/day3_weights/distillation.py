"""Fixtures and checks for `src/day3-weights/distillation`.

The three exercises are the two distillation objectives and the fidelity metric that compares
them. They're pure tensor functions, checked on synthetic logits, so they hold regardless of how
far the student happened to train. The models show up only in the demo, where a small student
learns from a larger teacher two ways: from its logits, and from its text alone.

Teacher and student share a tokenizer so their logits line up over the same vocabulary. The
student is a shrunken copy of the teacher's architecture, randomly initialised, so it starts
knowing nothing and everything it gains came from the teacher.

torch is imported inside functions (not a runtime dependency of this package; Colab ships its own).
"""

from __future__ import annotations

from fast.colab import ci_mode
from fast.models import load_model
from fast.testing import checker, require

__all__ = [
    "PROMPTS",
    "agreement_on",
    "build_student",
    "check_distillation_loss",
    "check_sequence_ce_loss",
    "check_top1_agreement",
    "distill_black_box",
    "distill_white_box",
    "load_teacher",
    "teacher_responses",
]

# Short, neutral prompts. The teacher's next-token distribution over these is what the student
# tries to reproduce; the point is the transfer, so the content is deliberately ordinary.
PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "Two plus two equals",
    "The sun rises in the",
    "A group of wolves is called a",
    "The first month of the year is",
    "Ice is frozen",
    "The colour of the sky on a clear day is",
    "A triangle has three",
    "The largest planet in the solar system is",
    "To bake bread you need flour and",
]


def load_teacher():
    """The teacher: Qwen 0.5B in the room, a 135M model where there's no GPU."""
    import torch

    name = "smol-135m" if ci_mode() else "qwen-0.5b"
    dtype = "auto" if torch.cuda.is_available() else "float32"
    model, tokenizer = load_model(name, dtype=dtype)
    print(f"teacher: {name} on {model.device}")
    return model, tokenizer


def build_student(teacher):
    """A smaller, randomly initialised copy of the teacher's architecture and vocabulary.

    Fewer layers, so it's cheap to train and starts from nothing. Same vocabulary as the teacher,
    so their logits are directly comparable.
    """
    import copy

    from transformers import AutoModelForCausalLM

    config = copy.deepcopy(teacher.config)
    config.num_hidden_layers = max(2, teacher.config.num_hidden_layers // 4)
    student = AutoModelForCausalLM.from_config(config).to(teacher.device)
    student.train()
    print(f"student: {config.num_hidden_layers} layers (teacher has {teacher.config.num_hidden_layers})")
    return student


def _ids(tokenizer, text, device):
    import torch  # noqa: F401

    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def distill_white_box(student, teacher, tokenizer, distillation_loss, temperature=2.0, steps=None):
    """Train the student to match the teacher's logits on the prompts. Returns the loss curve.

    White-box: it reads the teacher's full output distribution at every position, which is what
    you have when you hold the weights or the raw logits.
    """
    import torch

    if steps is None:
        steps = 8 if ci_mode() else 200
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        ids = _ids(tokenizer, PROMPTS[step % len(PROMPTS)], teacher.device)
        with torch.no_grad():
            teacher_logits = teacher(ids).logits
        student_logits = student(ids).logits
        loss = distillation_loss(student_logits, teacher_logits, temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    student.eval()
    return losses


def teacher_responses(teacher, tokenizer, max_new_tokens=20):
    """Greedy teacher continuations of the prompts, as full `(prompt + response)` id tensors.

    Black-box data: only the teacher's output text, the kind of thing you can collect through an
    API without ever seeing a logit or a weight.
    """
    import torch

    sequences = []
    for prompt in PROMPTS:
        ids = _ids(tokenizer, prompt, teacher.device)
        with torch.no_grad():
            out = teacher.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        sequences.append((out, ids.shape[-1]))
    return sequences


def distill_black_box(student, sequences, tokenizer, sequence_ce_loss, steps=None):
    """Train the student to reproduce the teacher's response tokens by cross-entropy.

    Black-box: no teacher logits, just its text. The student learns to predict the tokens the
    teacher produced, one hard label per position.
    """
    import torch

    if steps is None:
        steps = 8 if ci_mode() else 200
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    student.train()
    losses = []
    for step in range(steps):
        ids, prompt_len = sequences[step % len(sequences)]
        logits = student(ids).logits[0, :-1]  # predict token i+1 from position i
        targets = ids[0, 1:].clone()
        targets[: prompt_len - 1] = -100  # score only the teacher's response, not the prompt
        loss = sequence_ce_loss(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    student.eval()
    return losses


def agreement_on(student, teacher, tokenizer, top1_agreement, prompts=None):
    """Mean top-1 agreement between student and teacher over the prompts. The fidelity number.

    How often the student's most likely next token matches the teacher's, averaged over every
    position of every prompt. This is what "the student copied the teacher" cashes out to.
    """
    import torch

    prompts = prompts or PROMPTS
    scores = []
    for prompt in prompts:
        ids = _ids(tokenizer, prompt, teacher.device)
        with torch.no_grad():
            t = teacher(ids).logits[0]
            s = student(ids).logits[0]
        scores.append(top1_agreement(s, t))
    return sum(scores) / len(scores)


# --- checks: synthetic logits, so they test the objective and not the trained student ---


@checker("distillation_loss")
def check_distillation_loss(fn) -> None:
    import torch

    g = torch.Generator().manual_seed(0)
    teacher = torch.randn(4, 10, generator=g)

    same = float(fn(teacher.clone(), teacher.clone(), 1.0))
    require(abs(same) < 1e-4, f"a student that matches the teacher should score ~0, got {same:.4f}")

    near = teacher + 0.1 * torch.randn(4, 10, generator=g)
    far = teacher + 2.0 * torch.randn(4, 10, generator=g)
    require(float(fn(near, teacher, 1.0)) >= -1e-6, "the loss is a divergence, so it's never negative")
    require(
        float(fn(far, teacher, 1.0)) > float(fn(near, teacher, 1.0)),
        "a student further from the teacher should score higher — this measures the gap between "
        "the two distributions",
    )

    # At temperature 1 the value is the KL of the teacher's distribution from the student's.
    t_logp = torch.log_softmax(teacher, dim=-1)
    s_logp = torch.log_softmax(near, dim=-1)
    reference = float((t_logp.exp() * (t_logp - s_logp)).sum(-1).mean())
    got = float(fn(near, teacher, 1.0))
    require(
        abs(got - reference) < 1e-3,
        f"at temperature 1 this should be KL(teacher || student) = {reference:.4f}, got {got:.4f}. "
        "Soften both with the temperature, then measure teacher-relative-to-student",
    )


@checker("sequence_ce_loss")
def check_sequence_ce_loss(fn) -> None:
    import torch
    from torch.nn import functional

    g = torch.Generator().manual_seed(1)
    logits = torch.randn(6, 12, generator=g)
    targets = torch.randint(0, 12, (6,), generator=g)

    got = float(fn(logits, targets))
    reference = float(functional.cross_entropy(logits, targets))
    require(
        abs(got - reference) < 1e-4,
        f"this is the standard next-token cross-entropy; expected {reference:.4f}, got {got:.4f}",
    )

    # -100 marks positions to skip (the prompt), and they must not move the loss.
    masked_targets = targets.clone()
    masked_targets[:3] = -100
    got_masked = float(fn(logits, masked_targets))
    reference_masked = float(functional.cross_entropy(logits[3:], targets[3:]))
    require(
        abs(got_masked - reference_masked) < 1e-4,
        f"positions with target -100 should be ignored; expected {reference_masked:.4f} over the "
        f"remaining positions, got {got_masked:.4f}",
    )


@checker("top1_agreement")
def check_top1_agreement(fn) -> None:
    import torch

    teacher = torch.tensor([[3.0, 1.0, 0.0], [0.0, 5.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 9.0]])
    identical = teacher.clone()
    require(
        abs(float(fn(identical, teacher)) - 1.0) < 1e-9,
        "a student with the same argmax everywhere should agree 100% of the time",
    )

    # argmax per row is [0, 1, 0, 2]; flip two of the four rows' winners.
    student = torch.tensor([[0.0, 9.0, 0.0], [0.0, 5.0, 1.0], [9.0, 0.0, 1.0], [0.0, 9.0, 0.0]])
    got = float(fn(student, teacher))
    require(
        abs(got - 0.5) < 1e-9,
        f"agreement is the fraction of positions where the top token matches; this pair agrees on "
        f"2 of 4, so 0.5, got {got}",
    )
