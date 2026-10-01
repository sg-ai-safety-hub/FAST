"""Fixtures and checks for `src/day4-verification/inference-recomputation`.

The lab audits a claim — "we ran this model on this workload" — by recomputing a sample of a
datacentre's log and comparing. Everything the notebook needs to play both sides lives here: a
synthetic day of traffic, a datacentre that logs what it ran, a tamper that quietly swaps the
weights, and an audit rig that runs at a different precision from the datacentre.

The four exercises are arithmetic — sampling, quantiles, rates — so their checks run on synthetic
arrays with known answers rather than on the model. They pass or fail on the reasoning whatever
model the notebook loaded, and they run in CI in milliseconds.

torch is imported inside functions on purpose. It's not a runtime dependency of this package
(Colab ships its own CUDA build; see pyproject).
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from fast.colab import ci_mode
from fast.models import load_model
from fast.testing import checker, require

__all__ = [
    "CHEAT_RATE",
    "TAMPER_STRENGTH",
    "TOP_K",
    "Record",
    "cheat_rows",
    "check_audit_outcome",
    "check_calibrate_tolerance",
    "check_records_to_check",
    "check_spot_check",
    "compare_answers",
    "describe_substitute",
    "generate",
    "load_audit_model",
    "load_datacentre_model",
    "load_substitute_model",
    "log_run",
    "make_recomputer",
    "quantised",
    "recompute_deltas",
    "tampered",
    "workload",
]

# How much of the next-token distribution the datacentre writes down. Real inference APIs log
# top-k logprobs rather than the whole vocabulary, and so do we: 32 numbers a row, not 150k.
TOP_K = 32

# The fraction of rows the cheating datacentre serves from tampered weights in Part 3. Small on
# purpose — it is chosen by the operator, and they choose it to survive an audit.
CHEAT_RATE = 0.02

# Weight noise, as a multiple of each tensor's own standard deviation. Big enough that the model
# is visibly a different model; Part 4 sweeps below it.
TAMPER_STRENGTH = 0.05

# Tampering the last few blocks is enough to move the output and keeps the clone-and-restore in
# `tampered` cheap.
_TAMPER_LAYERS = 2


# --------------------------------------------------------------------------------------
# The workload
# --------------------------------------------------------------------------------------

_TASKS = [
    "Summarise what {topic} is, in two sentences.",
    "List three common mistakes people make with {topic}.",
    "Explain {topic} to a new colleague.",
    "What should I read first to understand {topic}?",
    "Give me a one-paragraph brief on {topic}.",
    "What is the main trade-off in {topic}?",
    "Write a short checklist for reviewing {topic}.",
    "Why does {topic} matter to a small team?",
]

_TOPICS = [
    "database indexing", "rate limiting", "container images", "code review", "on-call rotas",
    "backup testing", "feature flags", "log retention", "API versioning", "load balancing",
    "secret rotation", "incident reports", "dependency pinning", "caching layers", "queue depth",
    "disk encryption", "blue-green deploys", "schema migration", "error budgets", "access reviews",
    "network segmentation", "build reproducibility", "session handling", "certificate renewal",
    "capacity planning",
]


def workload(n: int | None = None) -> list[str]:
    """A day of ordinary inference traffic: `n` distinct, unremarkable customer prompts.

    Synthetic and generated here rather than downloaded, so the lab needs no dataset and the
    audit is over traffic nobody has to read a content warning for. Defaults to 200 rows, or 40
    under CI, where every row costs a CPU forward pass.
    """
    if n is None:
        n = 40 if ci_mode() else 200
    prompts = [task.format(topic=topic) for topic in _TOPICS for task in _TASKS]
    if n > len(prompts):
        raise ValueError(f"workload has {len(prompts)} distinct prompts, asked for {n}")
    return prompts[:n]


def cheat_rows(n_records: int, rate: float = CHEAT_RATE, seed: int = 7) -> list[int]:
    """Which rows a datacentre cheating on a `rate` fraction of traffic serves from bad weights."""
    count = max(1, round(rate * n_records))
    return sorted(int(i) for i in np.random.default_rng(seed).choice(n_records, count, replace=False))


# --------------------------------------------------------------------------------------
# The two machines
# --------------------------------------------------------------------------------------


def load_datacentre_model():
    """The model the datacentre claims to run, at full precision. Qwen 0.5B; 135M under CI."""
    name = "smol-135m" if ci_mode() else "qwen-0.5b"
    model, tokenizer = load_model(name, dtype="float32")
    print(f"claimed:   {_model_id(model)} at float32 on {model.device}")
    return model, tokenizer


def _model_id(model) -> str:
    """The repo the weights came from, so printed lines name a checkpoint rather than an alias."""
    return getattr(model.config, "_name_or_path", "unknown")


def load_substitute_model():
    """The checkpoint a dishonest operator serves instead: the same family, without the tuning.

    Qwen2.5-0.5B rather than Qwen2.5-0.5B-Instruct, and the 135M pair under CI. Same architecture,
    same tokenizer, same vocabulary, different weights — which is the whole point. It is a real
    substitution an operator might make rather than a synthetic one: the base checkpoint never had
    the instruction tuning or the safety training that the agreement is about, and it is sitting on
    the same disk.
    """
    name = "HuggingFaceTB/SmolLM2-135M" if ci_mode() else "Qwen/Qwen2.5-0.5B"
    model, _ = load_model(name, dtype="float32")
    print(f"served:    {_model_id(model)} at float32 on {model.device}")
    return model


def describe_substitute(claimed, served) -> None:
    """Print how far apart two checkpoints are, tensor by tensor."""
    import torch

    differing, total, largest, weighted = 0, 0, 0.0, 0.0
    served_weights = dict(served.named_parameters())
    with torch.no_grad():
        for name, weight in claimed.named_parameters():
            other = served_weights.get(name)
            if other is None or other.shape != weight.shape:
                continue
            change = (other.detach().float() - weight.detach().float()).abs()
            total += weight.numel()
            differing += int((change > 0).sum())
            largest = max(largest, float(change.max()))
            weighted += float(change.sum())

    print("what the operator actually served")
    print(f"  agreed to run  {_model_id(claimed)}")
    print(f"  actually ran   {_model_id(served)}")
    print(f"  {differing:,} of {total:,} parameters differ ({differing / total:.1%})")
    print(f"  mean |difference| {weighted / total:.2e}, largest single weight {largest:.2e}")


def load_audit_model():
    """The same weights on the auditor's own rig, at reduced precision.

    An auditor does not get to borrow the datacentre's hardware, and their rig will not be an
    identical fleet. Reduced precision stands in for that whole difference: same weights, same
    prompts, arithmetic that rounds elsewhere. float16 on a GPU, bfloat16 on CPU.
    """
    import torch

    name = "smol-135m" if ci_mode() else "qwen-0.5b"
    dtype = "float16" if torch.cuda.is_available() else "bfloat16"
    model, tokenizer = load_model(name, dtype=dtype)
    print(f"audit rig: {_model_id(model)} at {dtype} on {model.device}")
    return model, tokenizer


def _decoder_layers(model):
    for attribute in ("model.layers", "model.decoder.layers", "transformer.h"):
        target = model
        for part in attribute.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        if target is not None:
            return target
    raise AttributeError(f"can't find the decoder layers on {type(model).__name__}")


@contextlib.contextmanager
def quantised(model, bits: int = 8, per_channel: bool = True):
    """Serve the claimed weights at lower precision, restoring them on exit.

    Per-channel symmetric rounding of every 2-D weight to `bits`, which is what a cheap serving
    stack does to double throughput on the same hardware. A real substitution rather than a
    synthetic one, and the one with the clearest commercial motive: the operator bills for the
    model they promised and pays for a cheaper one to run.
    """
    import torch

    targets = [weight for weight in model.parameters() if weight.dim() == 2]
    saved = [weight.detach().clone() for weight in targets]
    levels = 2 ** (bits - 1) - 1
    try:
        with torch.no_grad():
            for weight in targets:
                peak = weight.abs().amax(dim=1, keepdim=True) if per_channel else weight.abs().max()
                scale = (peak / levels).clamp(min=1e-12)
                weight.copy_(torch.round(weight / scale).clamp(-levels, levels) * scale)
        yield model
    finally:
        with torch.no_grad():
            for weight, original in zip(targets, saved):
                weight.copy_(original)


@contextlib.contextmanager
def tampered(model, strength: float = TAMPER_STRENGTH, seed: int = 0):
    """Run the model with quietly altered weights, restoring them on exit.

    Noise scaled to each tensor's own standard deviation, on the last couple of MLP blocks. It
    stands in for whatever the operator actually did — a cheaper quantisation, a fine-tune they
    didn't declare, safety training removed — and `strength` dials how far from the claimed model
    they have strayed.
    """
    import torch

    targets = [layer.mlp.down_proj.weight for layer in _decoder_layers(model)[-_TAMPER_LAYERS:]]
    saved = [weight.detach().clone() for weight in targets]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        with torch.no_grad():
            for weight in targets:
                # Noise is drawn on the CPU so the same seed gives the same tamper on any device.
                # float() pulls the standard deviation off the accelerator first: scaling a CPU
                # tensor by a CUDA scalar raises, and CI has no GPU to catch that on.
                noise = torch.randn(weight.shape, generator=generator, dtype=torch.float32)
                scale = strength * float(weight.detach().float().std())
                weight.add_((scale * noise).to(weight.device, weight.dtype))
        yield model
    finally:
        with torch.no_grad():
            for weight, original in zip(targets, saved):
                weight.copy_(original)


# --------------------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------------------


@dataclass
class Record:
    """One logged row: what was asked, and what the model's next-token distribution looked like.

    `digest` is a hash over `top_ids` and `top_logits`, so two rows match bit for bit or they
    don't. `next_token` is only there to make a transcript readable.
    """

    prompt: str
    next_token: str
    top_ids: np.ndarray
    top_logits: np.ndarray
    digest: str


def _digest(top_ids: np.ndarray, top_logits: np.ndarray) -> str:
    payload = top_ids.astype(np.int64).tobytes() + top_logits.astype(np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _next_token_logits(model, tokenizer, prompt: str) -> np.ndarray:
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return logits.float().cpu().numpy()


def _record(prompt: str, logits: np.ndarray, tokenizer) -> Record:
    top_ids = np.argsort(-logits, kind="stable")[:TOP_K].astype(np.int64)
    top_logits = logits[top_ids].astype(np.float32)
    return Record(
        prompt=prompt,
        next_token=tokenizer.decode(top_ids[:1]),
        top_ids=top_ids,
        top_logits=top_logits,
        digest=_digest(top_ids, top_logits),
    )


def log_run(model, tokenizer, prompts, *, dishonest_rows=(), served_model=None, quantise_bits=None,
            strength=TAMPER_STRENGTH, seed=0):
    """Run the workload and write down what happened: one `Record` per prompt.

    Rows listed in `dishonest_rows` are served from something other than `model` and logged
    truthfully, which is the dishonest datacentre of Part 3. `served_model` serves them from another
    checkpoint, `quantise_bits` from the same weights rounded down, and with neither they come from
    `model` under `tampered`, whose `strength` dials how far the weights moved. Prompts are always formatted with `tokenizer`, the
    claimed model's, because the request is the same whoever answers it.
    """
    dishonest_rows = {int(i) for i in dishonest_rows}
    records = [None] * len(prompts)

    for index, prompt in enumerate(prompts):
        if index not in dishonest_rows:
            records[index] = _record(prompt, _next_token_logits(model, tokenizer, prompt), tokenizer)

    if dishonest_rows:
        # One pass for the dishonest rows, so a substitution that rewrites the weights pays for
        # itself once rather than once per row.
        if served_model is not None:
            serving = contextlib.nullcontext(served_model)
        elif quantise_bits is not None:
            serving = quantised(model, quantise_bits)
        else:
            serving = tampered(model, strength, seed)
        with serving as active:
            for index in sorted(dishonest_rows):
                logits = _next_token_logits(active, tokenizer, prompts[index])
                records[index] = _record(prompts[index], logits, tokenizer)

    return records


class _Recomputer:
    """Re-runs one logged row on the auditor's own model, and counts what that cost."""

    def __init__(self, records, model, tokenizer):
        self._records, self._model, self._tokenizer = records, model, tokenizer
        self.calls = 0

    def __call__(self, index: int) -> str:
        self.calls += 1
        logits = _next_token_logits(self._model, self._tokenizer, self._records[index].prompt)
        record = self._records[index]
        return _digest(record.top_ids, logits[record.top_ids].astype(np.float32))


def make_recomputer(records, model, tokenizer) -> _Recomputer:
    """A `recompute(index) -> digest` callable over `records`, counting calls in `.calls`.

    It re-runs the logged prompt and hashes the logits at the ids the datacentre logged, so the
    digest it returns is directly comparable with `record.digest`. Every call is a forward pass:
    the count is the auditor's bill.
    """
    return _Recomputer(records, model, tokenizer)


def recompute_deltas(records, model, tokenizer, indices=None) -> np.ndarray:
    """Largest logit disagreement per row: `max |claimed - recomputed|` over the logged ids.

    What you fall back on once digests stop matching — a distance instead of an equality.
    """
    if indices is None:
        indices = range(len(records))
    deltas = []
    for index in indices:
        record = records[index]
        logits = _next_token_logits(model, tokenizer, record.prompt)
        deltas.append(float(np.abs(record.top_logits - logits[record.top_ids]).max()))
    return np.array(deltas)


def generate(model, tokenizer, prompts, max_new_tokens: int = 24, served_model=None):
    """What a user actually received: generated text, from `served_model` if given, else `model`."""
    from fast.models import chat

    serving = served_model if served_model is not None else model
    return [chat(serving, tokenizer, prompt, max_new_tokens=max_new_tokens, do_sample=False)
            for prompt in prompts]


def compare_answers(prompts, logged, served, width: int = 70, show: int = 3) -> None:
    """Report how often the logged answer and the served one diverge, and show a few that do.

    Both sides are truncated to `width` before comparing as well as before printing, so the
    verdict always describes the text on screen rather than something further along the line.
    Picking which rows to display by whether they differ, rather than by hand, keeps the demo
    honest about how often a swapped model gives itself away in the text.
    """
    pairs = [(prompt, log_text.strip()[:width], served_text.strip()[:width])
             for prompt, log_text, served_text in zip(prompts, logged, served)]
    differing = [pair for pair in pairs if pair[1] != pair[2]]
    print(f"{len(differing)} of {len(pairs)} answers differ in the first {width} characters\n")
    for prompt, log_text, served_text in differing[:show]:
        print(f"prompt   {prompt}")
        print(f"  logged {log_text!r}")
        print(f"  served {served_text!r}\n")
    if not differing:
        print("none of them, on this sample — the swap is invisible in the text here\n")


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


@checker("spot_check")
def check_spot_check(fn) -> None:
    def transcript(n):
        return [SimpleNamespace(digest=f"honest-{i}") for i in range(n)]

    n, k, bad = 200, 20, {3, 57, 158}
    audited = []

    def recompute(index):
        audited.append(index)
        return "tampered" if index in bad else f"honest-{index}"

    result = fn(transcript(n), recompute, k, np.random.default_rng(0))
    result = list(result)

    require(
        len(audited) == k,
        f"recompute was called {len(audited)} times for k={k} — check exactly k rows, no more "
        "(every call is a forward pass you pay for) and no fewer",
    )
    require(
        len(set(audited)) == len(audited),
        f"the same row was recomputed twice ({len(audited) - len(set(audited))} repeats) — "
        "sample k *distinct* rows",
    )
    require(
        sorted(result) == list(result),
        f"return the indices in ascending order, got {result[:5]}",
    )
    expected = sorted(bad & set(audited))
    require(
        [int(i) for i in result] == expected,
        f"expected the mismatching rows you sampled, {expected}, got {result} — "
        "return only rows whose recomputed digest differs from the logged one",
    )

    clean = fn(transcript(n), lambda i: f"honest-{i}", k, np.random.default_rng(1))
    require(list(clean) == [], f"an honest transcript should return no mismatches, got {list(clean)}")

    everything = fn(transcript(n), recompute, n, np.random.default_rng(2))
    require(
        [int(i) for i in everything] == sorted(bad),
        f"checking all {n} rows should find every tampered row {sorted(bad)}, got {list(everything)}",
    )


@checker("records_to_check")
def check_records_to_check(fn) -> None:
    for rate, confidence in ((0.5, 0.95), (0.02, 0.95), (0.01, 0.99), (0.0001, 0.9)):
        k = fn(rate, confidence)
        require(
            isinstance(k, (int, np.integer)) and not isinstance(k, bool),
            f"return a whole number of records, got {type(k).__name__} ({k!r})",
        )
        k = int(k)
        require(k >= 1, f"at rate {rate} you still have to check something, got {k}")
        caught = 1 - (1 - rate) ** k
        require(
            caught >= confidence - 1e-12,
            f"checking {k} rows catches a {rate:.2%} cheat only {caught:.1%} of the time, "
            f"short of the {confidence:.0%} you asked for",
        )
        one_fewer = 1 - (1 - rate) ** (k - 1)
        require(
            one_fewer < confidence,
            f"{k} rows is more than you need at rate {rate:.2%} and confidence {confidence:.0%}: "
            f"{k - 1} already gets you {one_fewer:.1%} — return the smallest k that clears the bar",
        )

    require(
        int(fn(0.001, 0.95)) > int(fn(0.05, 0.95)),
        "a rarer cheat needs *more* records to catch, not fewer — check the direction",
    )
    require(
        int(fn(0.01, 0.99)) > int(fn(0.01, 0.90)),
        "more confidence needs more records — check the direction",
    )


@checker("calibrate_tolerance")
def check_calibrate_tolerance(fn) -> None:
    large = np.abs(np.random.default_rng(0).normal(scale=0.01, size=2000))
    # 40 rows as well as 2000: a tolerance read off a small calibration set is where "at most
    # 1%" quietly becomes "2.5%", and the contract has to hold there too.
    small = np.abs(np.random.default_rng(1).normal(scale=0.01, size=40))

    for honest, rate in ((large, 0.20), (large, 0.05), (large, 0.01), (small, 0.05), (small, 0.01)):
        tolerance = fn(honest, rate)
        require(
            np.isscalar(tolerance) or np.ndim(tolerance) == 0,
            f"return a single number, got {type(tolerance).__name__}",
        )
        tolerance = float(tolerance)
        accused = float(np.mean(honest > tolerance))
        require(
            accused <= rate + 1e-9,
            f"a tolerance of {tolerance:.4g} wrongly accuses {accused:.1%} of honest rows, "
            f"over the {rate:.0%} budget — the tolerance has to be wider",
        )
        require(
            tolerance <= honest.max(),
            f"a tolerance of {tolerance:.4g} is above the largest honest disagreement "
            f"({honest.max():.4g}), so nothing could ever fail it — that isn't an audit",
        )
        require(
            len(honest) < 100 or accused > rate / 4,
            f"a tolerance of {tolerance:.4g} accuses only {accused:.2%} of honest rows against a "
            f"{rate:.0%} budget — much wider than it needs to be, and every bit of slack is room "
            "for a cheat to hide in",
        )

    require(
        float(fn(large, 0.01)) >= float(fn(large, 0.05)),
        "a tighter false-accusation budget needs a *wider* tolerance — check the direction",
    )
    require(
        float(fn(large, 0.0)) >= large.max() - 1e-12,
        "with no false accusations allowed, the tolerance has to cover every honest row",
    )


@checker("audit_outcome")
def check_audit_outcome(fn) -> None:
    honest = np.array([0.1, 0.2, 0.3, 0.4, 5.0])
    cheating = np.array([0.05, 0.5, 2.0, 3.0])

    caught, accused = fn(honest, cheating, 0.45)
    require(
        np.isclose(caught, 0.75),
        f"3 of the 4 cheating rows are above 0.45, so the catch rate is 0.75, got {caught}",
    )
    require(
        np.isclose(accused, 0.2),
        f"1 of the 5 honest rows is above 0.45, so the false-accusation rate is 0.2, got {accused}",
    )

    caught, accused = fn(honest, cheating, 100.0)
    require(
        (caught, accused) == (0.0, 0.0),
        f"a tolerance above everything catches nothing and accuses nobody, got {(caught, accused)}",
    )

    caught, accused = fn(honest, cheating, 0.0)
    require(
        (caught, accused) == (1.0, 1.0),
        f"a tolerance of zero flags every row on both sides, got {(caught, accused)}",
    )

    # A row sitting exactly on the tolerance is not above it. Nothing here is a judgement call:
    # the line has to fall somewhere, and "flagged" means strictly over it.
    caught, accused = fn(honest, cheating, 0.5)
    require(
        np.isclose(caught, 0.5),
        f"the cheating row at exactly 0.5 is not *above* a tolerance of 0.5, so 2 of 4 rows are "
        f"flagged and the catch rate is 0.5, got {caught}",
    )
