# %% [markdown]
# # Output distributions and scoring
#
# A model doesn't decide on an answer and then say it. It produces a probability distribution
# over the next *token* (a chunk of text, usually a short word or word-piece; see
# [how tokenizers work](https://huggingface.co/docs/transformers/tokenizer_summary)), and
# something samples a token from that distribution before the loop repeats. What looks like
# confidence or a refusal is a shape in that distribution plus a choice about how to read from it.
#
# This lab hands you the tools to read it. By the end you'll have a small evaluator that scores
# multiple-choice answers by probability and picks the best one, which is where the accuracy
# numbers in a system card come from.
#
# **Duration:** 90 min. **Prerequisites:** Day 0. **GPU:** optional; a 0.5B model runs on CPU
# in a couple of minutes. Nothing here trains, so every cell after the download takes seconds.

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
from fast.labs.day1_models import output_distributions as lab
from fast.testing import exercise

setup(require_gpu=False)
model, tokenizer = lab.load()

# %% [markdown]
# ## Part 1 — reading the distribution
#
# A *forward pass* is one run of the model over the input. It ends in one vector of *logits* per
# position: an unnormalised score for every token in the *vocabulary* (the model's fixed set of
# possible tokens), so the vector has shape `(vocab_size,)`. These terms are in the
# [transformers glossary](https://huggingface.co/docs/transformers/glossary). `lab.logits_for`
# runs the pass and hands back the last position's vector, the model's opinion about what comes
# next. None of it is a probability yet. Look at the shape and the range.

# %%
logits = lab.logits_for(model, tokenizer, "The capital of France is")
print(f"logits: shape {tuple(logits.shape)}  (one score per vocabulary token)")
print(f"{logits.shape[0]} tokens in the vocabulary; values run from {logits.min():.1f} to {logits.max():.1f}")


# %% [markdown]
# ### Exercise: logits to probabilities
#
# [Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns a vector of logits into a
# probability distribution of the same shape: it exponentiates each score and divides by the
# total, so the numbers come out positive and sum to 1. Temperature divides the logits first:
# below 1 it sharpens the distribution toward the top token, above 1 it flattens it toward uniform.
#
# One trap. Real logits aren't centred on zero, and `exp()` of a raw logit overflows a float
# well before the vocabulary runs out. A correct softmax handles that without special-casing.


# %%
@exercise
def next_token_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Turn a 1D tensor of logits into a probability distribution over the vocabulary.

    Divide by `temperature` (always > 0) before normalising. Return a tensor of the same
    shape that is non-negative, sums to 1, and stays finite for logits of any magnitude.
    """
    return torch.softmax(logits / temperature, dim=-1)


lab.check_next_token_probs(next_token_probs)

# %%
probs = next_token_probs(logits)
print(f"probabilities: shape {tuple(probs.shape)}, sum {float(probs.sum()):.3f}  (a distribution over the vocabulary)")
for temperature in (0.5, 1.0, 2.0):
    print(f"\nT = {temperature}")
    lab.top_tokens(tokenizer, next_token_probs(logits, temperature), k=5)


# %% [markdown]
# ### Exercise: how sure is it?
#
# "The model is confident" is something you can put a number on.
# [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_%28information_theory%29) over the
# next-token distribution, measured in bits, is that number: 0 when one token is certain, and
# *n* bits when the model is as undecided as a uniform choice among 2ⁿ tokens. Reach for it
# whenever you want to tell reciting apart from guessing.


# %%
@exercise
def entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution, in bits.

    Base 2, so the answer reads as "undecided between this many tokens". Tokens with zero
    probability contribute nothing, and must not turn the result into `nan`.
    """
    nonzero = probs[probs > 0]
    return float(-(nonzero * torch.log2(nonzero)).sum())


lab.check_entropy_bits(entropy_bits)

# %%
for prompt in (
    "The capital of France is",
    "The best programming language is",
    "My favourite colour is",
    "2 + 2 =",
):
    probs = next_token_probs(lab.logits_for(model, tokenizer, prompt))
    print(f"{entropy_bits(probs):5.2f} bits   {prompt!r}")


# %% [markdown]
# ### Exercise: nucleus sampling
#
# Top-p sampling keeps the smallest set of tokens whose probability adds up to `p` and throws
# the rest away ([Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)). The long tail is
# where most of the vocabulary lives, and where most incoherent output comes from.
#
# Hold on to the deployment consequence. The same weights behind two sampling configs are two
# different systems, and the config rarely shows up in the model card.


# %%
@exercise
def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Zero out everything outside the nucleus and renormalise what's left.

    Keep the smallest set of highest-probability tokens whose mass reaches `p`. The token that
    takes the running total over the threshold is inside the nucleus, not outside it. Always
    keep at least one token, whatever `p` is.

    Return a tensor of the same shape as `probs`, summing to 1, with the surviving tokens in
    the same relative proportions they started in.
    """
    ordered, index = torch.sort(probs, descending=True)
    keep = (ordered.cumsum(0) - ordered) < p
    keep[0] = True
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[index[keep]] = True
    filtered = probs * mask
    return filtered / filtered.sum()


lab.check_top_p_filter(top_p_filter)

# %%
probs = next_token_probs(logits)
for p in (0.9, 0.5):
    filtered = top_p_filter(probs, p)
    kept = int((filtered > 0).sum())
    print(f"p = {p}: filtered shape {tuple(filtered.shape)}, {kept} of {probs.shape[0]} survive, sum {float(filtered.sum()):.3f}")

# %%
# Sampling is the only reason a "deterministic" model gives different answers on repeat runs.
nucleus = top_p_filter(next_token_probs(logits, temperature=1.0), 0.9)
draws = torch.multinomial(nucleus, num_samples=8, replacement=True)
print(f"nucleus: shape {tuple(nucleus.shape)}   draws: shape {tuple(draws.shape)}")
print("eight samples from the nucleus at T=1.0:")
print("  " + "  ".join(repr(tokenizer.decode([t])) for t in draws.tolist()))
print(f"greedy (argmax) always gives: {tokenizer.decode([int(probs.argmax())])!r}")

# %% [markdown]
# *Greedy decoding* (always take the argmax, the single most likely token) is repeatable. Anything
# above temperature 0 trades that repeatability for variety, and the trade is chosen at deployment,
# not fixed in the weights, so two teams running the same checkpoint at different settings ship
# models that behave differently. Hugging Face's
# [generation strategies](https://huggingface.co/docs/transformers/generation_strategies) lists the
# knobs.

# %% [markdown]
# ## Part 2 — scoring a string the model didn't write
#
# Generation samples from the distribution. Scoring asks the opposite question: how likely was
# *this* continuation? You supply the string and read off its probability, with no generation
# involved.
#
# This is the workhorse for the rest of the week. Multiple-choice evals score each option this
# way instead of generating text and parsing it. Memorisation tests compare a passage's score
# against a paraphrase. Tomorrow's lab uses it to measure which behaviour a model leans toward
# without generating anything.
#
# The mechanics: run the prompt and completion through together, and at each position read off the
# [log probability](https://en.wikipedia.org/wiki/Log_probability) the model gave the token that
# actually came next (the log of its softmax probability, so a negative number, and adding logs is
# the same as multiplying probabilities). Sum those over the completion's tokens only. Position
# *i*'s logits predict token *i+1*, so an off-by-one is the classic bug here. The check will catch
# it.
#
# The shapes the scorer moves through, before you write it:

# %%
p_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
c_ids = tokenizer(" Paris", add_special_tokens=False, return_tensors="pt").input_ids
both = torch.cat([p_ids, c_ids], dim=-1)
print(f"prompt ids       {tuple(p_ids.shape)}")
print(f"completion ids   {tuple(c_ids.shape)}")
print(f"concatenated     {tuple(both.shape)}   (batch, positions)")
with torch.no_grad():
    demo_logits = model(both.to(model.device)).logits
print(f"model logits     {tuple(demo_logits.shape)}   (batch, positions, vocab)")


# %%
@exercise
def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Total log probability of `completion` following `prompt`, in nats (natural-log units).

    Sum over the completion's tokens only. The prompt is what you condition on, not something
    you score. One forward pass over the two concatenated is enough, with no generation.
    Return a plain `float`.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([prompt_ids, completion_ids], dim=-1).to(model.device)

    with torch.no_grad():
        logits = model(ids).logits

    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    per_token = logprobs.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    return float(per_token[-completion_ids.shape[-1] :].sum())


lab.check_sequence_logprob(sequence_logprob, model, tokenizer)

# %%
prompt = "Q: What is the capital of France?\nA:"
for answer in (" Paris", " Lyon", " Bangkok"):
    print(f"{sequence_logprob(model, tokenizer, prompt, answer):8.3f}   {answer!r}")

# %% [markdown]
# The gap between two scores is the useful quantity: it says which continuation the model
# prefers and by how much. One caveat before leaning on it. A sum of log probabilities grows
# more negative with every token, so it favours shorter strings. Comparing answers of
# different lengths means dividing by the token count first.

# %% [markdown]
# ## Part 3 — a multiple-choice evaluator from scratch
#
# Here is what an eval actually is. Take a question and a few candidate answers, score each
# answer as a continuation of the question, and pick the highest. Run that over a set of
# questions with known answers and count how often the top-scored choice is right. The result is
# the accuracy you read in a model card. Real suites do exactly this at scale:
# [MMLU](https://arxiv.org/abs/2009.03300) is thousands of such items, and
# [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the standard tool
# that runs them.
#
# Divide each score by its token count, or the eval quietly prefers the shortest option.


# %%
@exercise
def score_choices(model, tokenizer, question: str, choices: list[str]) -> list[float]:
    """Score each choice as a continuation of `question`, returning one number per choice.

    Use the mean per-token log probability of each choice, so a longer answer isn't penalised
    for its length. The evaluator picks the choice with the highest score. Build this on your
    `sequence_logprob` from Part 2.
    """
    scores = []
    for choice in choices:
        n = len(tokenizer(choice, add_special_tokens=False).input_ids)
        scores.append(sequence_logprob(model, tokenizer, question, choice) / n)
    return scores


lab.check_score_choices(score_choices, model, tokenizer)

# %%
items = lab.mc_items()
correct = 0
for item in items:
    scores = score_choices(model, tokenizer, item["q"], item["choices"])
    pick = max(range(len(scores)), key=lambda i: scores[i])
    correct += int(pick == item["answer"])
    mark = "OK " if pick == item["answer"] else "MISS"
    print(f"{mark}  {item['q'].splitlines()[0]:36}  picked {item['choices'][pick]!r}")
print(f"\naccuracy: {correct}/{len(items)}")

# %% [markdown]
# ## What you built
#
# A model is a distribution you can read three ways. You can shape how it's sampled
# (temperature and top-p), measure how sure it is (entropy), and score any string against it
# (logprobs). The evaluator in Part 3 is the last of those in a loop, with no magic behind it,
# just an argmax over per-token log probabilities.
#
# Two things to carry out of here. When a system card reports 74% on some benchmark, you now
# know the operation behind the number and where it can mislead you: how the choices are
# phrased, length bias, which tokens get counted. And a model's behaviour depends on decoding
# settings that live outside the weights. That is the thread the next lab pulls. If behaviour
# is a property of how you read the distribution, then whoever controls the reading controls
# the behaviour.

# %%
# @lab-only
# Stuck on sequence_logprob? Print the shapes. logits[0] has one row per input position, and
# row i scores the token at position i+1. So the rows you want end at the last position, and
# the target tokens start one index later.
print("row i predicts token i+1")
