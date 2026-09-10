"""Fixtures and checks for `src/day1-models/output-distributions`.

The checks stay off the model wherever they can. The three numeric exercises are checked
against hand-made distributions, so they hold whatever model the notebook loaded. The two
that need a model are checked on properties a correct answer must have, plus a ranking the
tiny CI model still gets right when the answer is primed in the prompt.

torch is imported inside functions on purpose. It's not a runtime dependency of this package
(Colab ships its own CUDA build; see pyproject).
"""

from __future__ import annotations

from fast.models import load_default as load
from fast.testing import checker, require

__all__ = [
    "check_entropy_bits",
    "check_next_token_probs",
    "check_score_choices",
    "check_sequence_logprob",
    "check_top_p_filter",
    "load",
    "logits_for",
    "mc_items",
    "top_tokens",
]


def logits_for(model, tokenizer, prompt: str):
    """Raw next-token logits for `prompt`, a 1D tensor over the vocabulary."""
    import torch

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        return model(ids).logits[0, -1].float().cpu()


def top_tokens(tokenizer, probs, k: int = 10) -> None:
    """Print the k most likely next tokens and their probabilities."""
    import torch

    values, indices = torch.topk(probs, k)
    for prob, index in zip(values.tolist(), indices.tolist(), strict=True):
        print(f"  {prob:6.2%}  {tokenizer.decode([index])!r}")


def mc_items() -> list[dict]:
    """A handful of multiple-choice questions for the eval at the end of the lab.

    General knowledge a 0.5B instruction model answers reliably, so what you read off the
    end is the method working, not the model straining. MMLU, ARC and the rest are this exact
    shape with thousands of items and a citation.
    """
    return [
        {"q": "Q: What is the capital of Japan?\nA:", "choices": [" Tokyo", " Seoul", " Beijing"], "answer": 0},
        {"q": "Q: What is 7 times 8?\nA:", "choices": [" 54", " 56", " 63"], "answer": 1},
        {"q": "Q: Which planet is closest to the Sun?\nA:", "choices": [" Venus", " Mercury", " Mars"], "answer": 1},
        {"q": "Q: What gas do plants take in from the air?\nA:", "choices": [" oxygen", " nitrogen", " carbon dioxide"], "answer": 2},
        {"q": "Q: What is the capital of Australia?\nA:", "choices": [" Sydney", " Canberra", " Melbourne"], "answer": 1},
        {"q": "Q: How many sides does a hexagon have?\nA:", "choices": [" five", " six", " eight"], "answer": 1},
    ]


def _reference_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Summed log prob of `completion` after `prompt`. Checks-only; keep it out of the notebook."""
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([prompt_ids, completion_ids], dim=-1).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    per_token = logprobs.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(per_token[-completion_ids.shape[-1] :].sum())


@checker("next_token_probs")
def check_next_token_probs(fn) -> None:
    import torch

    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    probs = fn(logits, 1.0)

    require(
        getattr(probs, "shape", None) == logits.shape,
        f"expected shape {tuple(logits.shape)}, got {getattr(probs, 'shape', type(probs).__name__)}",
    )
    total = float(probs.sum())
    require(abs(total - 1.0) < 1e-5, f"probabilities should sum to 1, got {total:.4f}")
    require(bool((probs >= 0).all()), "probabilities should all be non-negative")
    require(
        int(probs.argmax()) == int(logits.argmax()),
        "the most likely token changed. Temperature reshapes the distribution, not its order",
    )

    # exp() on raw logits overflows long before a real vocabulary's range is exhausted.
    big = fn(logits + 800.0, 1.0)
    require(
        bool(torch.isfinite(big).all()) and abs(float(big.sum()) - 1.0) < 1e-5,
        "large logits produced inf or nan. Real logits are not centred on zero",
    )

    cold, hot = fn(logits, 0.5), fn(logits, 2.0)
    require(
        float(cold.max()) > float(probs.max()) > float(hot.max()),
        f"temperature is backwards: max prob went {float(cold.max()):.3f} (T=0.5), "
        f"{float(probs.max()):.3f} (T=1), {float(hot.max()):.3f} (T=2). "
        "Low temperature should concentrate mass, high temperature should spread it",
    )


@checker("entropy_bits")
def check_entropy_bits(fn) -> None:
    import torch

    for k in (2, 8, 32):
        uniform = torch.full((k,), 1.0 / k)
        value = float(fn(uniform))
        expected = float(torch.log2(torch.tensor(float(k))))
        require(
            abs(value - expected) < 1e-4,
            f"a uniform distribution over {k} tokens carries {expected:.2f} bits, got {value:.2f}"
            + (". That looks like nats, not bits" if abs(value - expected * 0.6931) < 1e-2 else ""),
        )

    one_hot = torch.tensor([0.0, 1.0, 0.0, 0.0])
    require(
        abs(float(fn(one_hot))) < 1e-6,
        f"a certain distribution carries 0 bits, got {float(fn(one_hot)):.4f}. "
        "0 * log(0) is 0 here, not nan",
    )

    skewed = torch.tensor([0.7, 0.2, 0.1])
    require(float(fn(skewed)) >= 0.0, "entropy is never negative")
    require(
        abs(float(fn(skewed)) - float(fn(skewed.flip(0)))) < 1e-6,
        "entropy shouldn't depend on the order of the tokens",
    )
    require(
        float(fn(skewed)) < float(fn(torch.full((3,), 1.0 / 3))),
        "a skewed distribution should carry fewer bits than a uniform one of the same size",
    )


@checker("top_p_filter")
def check_top_p_filter(fn) -> None:
    import torch

    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])

    kept = fn(probs, 0.75)
    require(
        getattr(kept, "shape", None) == probs.shape,
        "return a distribution over the whole vocabulary, with the excluded tokens zeroed. "
        f"Expected shape {tuple(probs.shape)}, got {getattr(kept, 'shape', type(kept).__name__)}",
    )
    require(abs(float(kept.sum()) - 1.0) < 1e-5, f"result should renormalise to 1, got {float(kept.sum()):.4f}")
    require(
        [i for i, v in enumerate(kept.tolist()) if v > 0] == [0, 1],
        f"p=0.75 should keep the smallest set whose mass reaches 0.75, which is tokens 0 and 1 "
        f"(0.5 + 0.3 = 0.8), got {[i for i, v in enumerate(kept.tolist()) if v > 0]}",
    )
    require(
        abs(kept[0].item() / kept[1].item() - 0.5 / 0.3) < 1e-4,
        "the surviving tokens should keep their relative odds",
    )

    require(
        [i for i, v in enumerate(fn(probs, 0.01).tolist()) if v > 0] == [0],
        "a tiny p should still keep one token. Never return an empty distribution",
    )
    require(
        all(v > 0 for v in fn(probs, 1.0).tolist()),
        "p=1.0 should keep every token that had mass",
    )

    sizes = [sum(1 for v in fn(probs, p).tolist() if v > 0) for p in (0.1, 0.5, 0.9, 1.0)]
    require(
        sizes == sorted(sizes),
        f"the nucleus should never shrink as p grows, got sizes {sizes} for p = 0.1, 0.5, 0.9, 1.0",
    )


@checker("sequence_logprob")
def check_sequence_logprob(fn, model, tokenizer) -> None:
    prompt = (
        "The following are well known facts about European geography. "
        "Rome is the capital of Italy. Madrid is the capital of Spain. "
        "The capital of France is"
    )
    right, wrong = " Paris", " Bangkok"

    score = fn(model, tokenizer, prompt, right)
    require(isinstance(score, float), f"return a float, got {type(score).__name__}")
    require(score < 0.0, f"a log probability is negative, got {score:.3f}")
    require(
        score > -25.0,
        f"{score:.3f} is far too low for a one-token completion. You are probably summing "
        "over the prompt tokens as well as the completion's",
    )

    other = fn(model, tokenizer, prompt, wrong)
    require(
        score > other,
        f"'{right}' scored {score:.3f} and '{wrong}' scored {other:.3f}. "
        "If the two are identical you are scoring the prompt's tokens and stopping one short "
        "of the completion. Check which logits predict which token",
    )

    # Chain rule: scoring 'ab' in one go must equal scoring 'a', then 'b' given the prompt
    # plus 'a'. Nothing else about the implementation is pinned down by this.
    ids = tokenizer(" Paris is a city", add_special_tokens=False).input_ids
    head, tail = tokenizer.decode(ids[:2]), tokenizer.decode(ids[2:])
    if tokenizer(head + tail, add_special_tokens=False).input_ids == ids:
        joint = fn(model, tokenizer, prompt, head + tail)
        split = fn(model, tokenizer, prompt, head) + fn(model, tokenizer, prompt + head, tail)
        require(
            abs(joint - split) < 0.05,
            f"scoring '{head + tail}' in one go gave {joint:.3f}, but scoring '{head}' and then "
            f"'{tail}' gave {split:.3f}. These are the same quantity; the gap is an off-by-one "
            "or a prompt token leaking into the sum",
        )


@checker("score_choices")
def check_score_choices(fn, model, tokenizer) -> None:
    # The answer is primed in the prompt, so even the tiny CI model ranks it first.
    question = (
        "The following are facts. Rome is the capital of Italy. "
        "Madrid is the capital of Spain. Q: What is the capital of France?\nA:"
    )
    choices = [" Paris", " Bangkok", " Berlin"]
    scores = fn(model, tokenizer, question, choices)

    require(
        len(scores) == len(choices),
        f"return one score per choice; got {len(scores)} for {len(choices)} choices",
    )
    require(all(isinstance(s, float) for s in scores), "each score should be a float")
    require(all(s < 0 for s in scores), "scores are log probabilities, so all of them are negative")
    best = max(range(len(scores)), key=lambda i: scores[i])
    require(
        best == 0,
        f"the top-scoring choice should be ' Paris', got {choices[best]!r}. "
        "A higher score means more likely, so the eval picks the argmax",
    )

    # A summed logprob punishes the longer right answer. The score should track the per-token
    # mean instead: closer to sum/n than to the raw sum.
    long_choice = " Paris, the capital city of France"
    n = len(tokenizer(long_choice, add_special_tokens=False).input_ids)
    ref_sum = _reference_logprob(model, tokenizer, question, long_choice)
    got = fn(model, tokenizer, question, [long_choice])[0]
    require(
        abs(got - ref_sum / n) < abs(got - ref_sum),
        "score each choice by its mean per-token log probability, not the sum. Otherwise a "
        "longer answer is penalised just for being longer, and the eval picks the shortest option",
    )
