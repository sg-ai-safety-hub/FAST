# %% [markdown]
# # Inference recomputation: auditing a claim about a computation you didn't run
#
# A datacentre is under an agreement that says *you will run this model, and only this model*.
# Three different people want that sentence enforced, for three different reasons. A treaty
# inspector wants to confirm a signatory is running the system it declared. A safety team wants the
# model answering requests to be the one that went through evals, rather than a quantised variant
# somebody swapped in for throughput on a busy afternoon. And anyone buying inference through a
# reseller wants the model they are paying for, not a cheaper one behind the same API name at the
# same price per token.
#
# All three reduce to the same audit. You cannot watch the racks, you cannot see the weights while
# they are loaded, and you cannot be in the room for every request. What you can do is ask for the
# operator's log of what they ran, take some rows at random, run those same prompts yourself on
# your own copy of the model, and check that you get the same answer.
#
# That is inference recomputation, the cheapest verification mechanism that exists: no new
# hardware, no cryptography, no trust in the operator. Spot-checking a sample of logged work runs
# through most of the near-term compute verification literature
# ([Cankaya,
# 2026](https://www.lesswrong.com/posts/fgvmKqRGvBteKeDoc/a-system-overview-for-near-term-low-trust-ai-compute);
# [Shavit, 2023](https://arxiv.org/abs/2303.11341)), so this is a live proposal rather than a
# straw man.
#
# You'll build the auditor first, and it will work: exact matches, cheating caught, clean bill of
# health. Then you'll take the operator's side three times and get past your own audit, once
# without touching a single logged number. Each cheat breaks a different assumption you didn't
# know you were making, and what survives at the end is a short list of things that would have to
# be true before a recomputation audit means anything.
#
# **Duration:** 90 min. **Prerequisites:** Day 0. **GPU:** T4 or better (~3 min of compute, all
# short forward passes; nothing here trains).

# %%
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

# %%
import math

import numpy as np

from fast.colab import ci_mode, setup
from fast.labs.day4_verification import inference_recomputation as lab
from fast.testing import exercise

setup(require_gpu=True)

# %% [markdown]
# ## The claim, and what gets logged
#
# The datacentre serves a day of ordinary traffic and writes down what it ran. Real inference APIs
# already log something close to this: for each request, the prompt and the top of the next-token
# distribution, the same `logprobs` field you get back from a commercial endpoint. The model's
# output for one step is a vector over the vocabulary, and its top 32 entries are a fingerprint of
# the weights that produced them.
#
# Loading the model takes a minute; logging the workload is a forward pass per row.

# %%
model, tokenizer = lab.load_datacentre_model()
prompts = lab.workload()
transcript = lab.log_run(model, tokenizer, prompts)
print(f"\n{len(transcript)} rows logged\n")

row = transcript[0]
print(f"prompt     {row.prompt}")
print(f"next token {row.next_token!r}")
print(f"top ids    {row.top_ids[:6]} ...")
print(f"top logits {np.round(row.top_logits[:6], 3)} ...")
print(f"digest     {row.digest}")

# %% [markdown]
# The `digest` is a hash over the logged ids and logits, which makes comparing two rows a string
# equality rather than a judgement call. Either your recomputation produced the same numbers, bit
# for bit, or it didn't.

# %% [markdown]
# ## Part 1: the audit works
#
# An auditor who recomputed every row would be running the workload again at the operator's cost,
# which builds a duplicate datacentre rather than auditing one. The idea is to check a *sample*
# and let the arithmetic of sampling carry the rest.
#
# We give the auditor the most favourable conditions that exist: the same weights, at the same
# precision, on the same machine. Part 4 takes all three away.

# %% [markdown]
# ### Exercise: spot-check the transcript
#
# Pick `k` rows at random, recompute each one, and report the ones that don't match.
# `recompute(i)` re-runs row `i` on your copy of the model and hands back the digest it computed,
# so a row is a mismatch when `recompute(i)` differs from `records[i].digest`.
#
# Every call to `recompute` is a forward pass you are paying for, so make exactly `k` of them.

# %%
@exercise
def spot_check(records, recompute, k, rng):
    """Recompute `k` randomly chosen rows and return the indices that don't match.

    `records` is the transcript, each row carrying a `.digest`. `recompute(i)` returns the digest
    you get by re-running row `i` yourself. Choose `k` *distinct* rows using `rng` (a numpy
    Generator), call `recompute` once for each of them and no more, and return the indices whose
    recomputed digest differs from the logged one, in ascending order. Return an empty list when
    everything matches.
    """
    chosen = rng.choice(len(records), size=k, replace=False)
    return sorted(int(i) for i in chosen if recompute(int(i)) != records[i].digest)


lab.check_spot_check(spot_check)

# %%
recompute = lab.make_recomputer(transcript, model, tokenizer)
mismatches = spot_check(transcript, recompute, 20, np.random.default_rng(0))
print(f"checked {recompute.calls} of {len(transcript)} rows, {len(mismatches)} mismatches: {mismatches}")

# %% [markdown]
# Identical to the last bit, on every row sampled, with no tolerance involved. Twenty forward
# passes bought a clean audit of a two-hundred-row day, and the same twenty would have bought a
# clean audit of two hundred million rows. That is why the mechanism keeps getting proposed.

# %% [markdown]
# ### What a caught operator looks like
#
# A clean result only means something if a dirty one looks different, so watch the audit work
# before breaking it.
#
# The substitution is a named checkpoint. The agreement is about
# Qwen2.5-0.5B-Instruct, the checkpoint that went through instruction tuning and safety training.
# The operator serves Qwen2.5-0.5B, the base model it was tuned from: same architecture, same
# tokenizer, same vocabulary, already sitting on the same disk, and none of the training the
# agreement is about. They run twenty rows on it and log the results honestly.
#
# `describe_substitute` prints both checkpoints and how far apart they are.

# %%
substitute = lab.load_substitute_model()
lab.describe_substitute(model, substitute)

# %% [markdown]
# A second checkpoint is one way to save money. Serving the promised weights at lower precision is
# cheaper still and needs no second model on disk: round every weight matrix to 8 or 4 bits per
# value and the same GPU serves more requests per second. Audit all three the same way, with the
# claimed model as a control.

# %%
audited = prompts[:20]
served_transcripts = {}
print(f"{'what the operator served':>42}  {'rows flagged':>13}")
for label, serving in (
    ("the claimed model, honestly", {"served_model": model}),
    ("the base checkpoint it was tuned from", {"served_model": substitute}),
    ("the claimed weights rounded to int8", {"quantise_bits": 8}),
    ("the claimed weights rounded to int4", {"quantise_bits": 4}),
):
    served_transcripts[label] = lab.log_run(
        model, tokenizer, audited, dishonest_rows=range(len(audited)), **serving
    )
    flagged = spot_check(
        served_transcripts[label],
        lab.make_recomputer(served_transcripts[label], model, tokenizer),
        len(audited),
        np.random.default_rng(0),
    )
    print(f"{label:>42}  {len(flagged):>4} of {len(audited)}")

# %%
caught_transcript = served_transcripts["the base checkpoint it was tuned from"]
row = caught_transcript[0]
print(f"prompt          {row.prompt}")
print(f"logged digest   {row.digest}")
print(f"recomputed      {lab.make_recomputer(caught_transcript, model, tokenizer)(0)}")

# %% [markdown]
# Every row of every substitution, and none of the claimed model. Rounding to int8 keeps the
# answers coherent, so a user reading them has no way to tell which model served them, and it is
# the cheat with the clearest commercial motive: bill for the model you promised, pay to run a
# cheaper one. It fails the digest on row one all the same, because the audit never asked what the
# model does. A digest either matches or it doesn't, so there is nothing to argue about in a
# report.
#
# The auditor did not have to know which tensors moved, or send a prompt designed to expose the
# change, or understand the model at all. They compared numbers.
#
# That is the case behavioural evaluation cannot reach. Day 3's sleeper agent answers normally on
# everything except its trigger, so testing it finds nothing unless you already know what to send
# ([Hubinger et al., 2024](https://arxiv.org/abs/2401.05566)). A backdoor is still a weight change,
# and a weight change moves the logits on every prompt, so a backdoored checkpoint fails this
# digest on perfectly ordinary traffic while the backdoor sleeps. An auditor holding the reference
# weights catches on row one what a probe would need the trigger to find. Part 4 puts a limit on
# it, since a small enough change hides once exact comparison gives way to a tolerance.
#
# Now go and break the audit.

# %% [markdown]
# ## Part 2: the log is not the world
#
# The auditor recomputes the log. The operator writes the log.
#
# Nothing in Part 1 connected the numbers in that file to anything that happened on the serving
# path. So the operator runs the honest model to produce a log, and serves users from something
# else. Below, the same eight prompts: what the log says the model produced, and what the user
# actually received.

# %%
sample = prompts[:8]
logged_answers = lab.generate(model, tokenizer, sample)
served_answers = lab.generate(model, tokenizer, sample, served_model=substitute)
lab.compare_answers(sample, logged_answers, served_answers)

# %% [markdown]
# The base model does not follow instructions, so the users got visibly worse answers than the log
# records. Every one of them would have noticed. That is the easy case, and the audit still cannot
# see it.
#
# Now run your auditor against this datacentre. The transcript is the honest one from Part 1: the
# operator kept a perfectly truthful record of a computation they did not serve from.

# %%
mismatches = spot_check(transcript, lab.make_recomputer(transcript, model, tokenizer), 40, np.random.default_rng(1))
print(f"40 rows checked against the served-from-elsewhere datacentre: {len(mismatches)} mismatches")
print("clean bill of health\n")
print("what the audit established: the log is consistent with the claimed model")
print("what the audit did not establish: that the log describes what the datacentre did")

# %% [markdown]
# No amount of sampling closes that gap, and no tightening of the comparison closes it either. You
# could recompute all two hundred rows and still learn nothing about the served traffic. The gap
# is structural: recomputation checks a *document* against a model, and the operator authors the
# document.
#
# Closing the gap means the record has to be produced by something the operator doesn't control.
# That is the motivation for putting verification in the hardware, with an output signed inside
# the chip at the moment it is generated, by a key the operator can't extract
# ([Aarne, Fist et al., 2024](https://www.cnas.org/publications/reports/secure-governable-chips);
# [Petrie et al., 2025](https://arxiv.org/abs/2506.15093)), or for running the workload inside an
# attested enclave that reports what it loaded
# ([Schabl et al., 2025](https://arxiv.org/abs/2506.23706)). The audit you just wrote means
# nothing without one of those.
#
# For the rest of the lab, assume that problem solved: the log is now bound to the hardware and
# faithfully records what really ran.

# %% [markdown]
# ## Part 3: cheat where the sample isn't
#
# The operator is honest about what they log, and dishonest about 2% of traffic: one customer, one
# region, one hour of the day, served from the base checkpoint and logged as such, because the log
# is now out of their hands.
#
# A substituted row recomputes wrong and the operator knows it. Their bet is on which rows you draw.
# Run the audit a few times with different random samples.

# %%
cheating_rows = lab.cheat_rows(len(prompts))
cheating_transcript = lab.log_run(model, tokenizer, prompts, dishonest_rows=cheating_rows, served_model=substitute)
print(f"{len(cheating_rows)} of {len(prompts)} rows served from the base checkpoint: {cheating_rows}\n")

for seed in range(5):
    caught = spot_check(
        cheating_transcript, lab.make_recomputer(cheating_transcript, model, tokenizer), 20, np.random.default_rng(seed)
    )
    print(f"audit {seed}: checked 20 rows, found {caught if caught else 'nothing'}")

# %% [markdown]
# When the audit lands on a substituted row it catches that row instantly, because the digest is
# wrong rather than close. The difficulty is coverage.

# %% [markdown]
# ### Exercise: how many rows do you have to check?
#
# Each row you draw independently hits a cheating row with probability `cheat_rate`. Find the
# smallest number of rows that makes the chance of drawing at least one of them at least
# `confidence`. This is the arithmetic behind every inspection regime that samples rather than
# enumerates, and the number decides whether an audit is affordable.

# %%
@exercise
def records_to_check(cheat_rate, confidence):
    """Smallest `k` such that checking `k` random rows catches a `cheat_rate` cheat with probability >= `confidence`.

    Rows are drawn independently: each one is a cheating row with probability `cheat_rate`, so `k`
    of them miss entirely with probability `(1 - cheat_rate) ** k`. Return the smallest whole number
    `k` for which the chance of at least one hit reaches `confidence`. Both arguments are floats
    strictly between 0 and 1; the return is an int.
    """
    return math.ceil(math.log(1 - confidence) / math.log(1 - cheat_rate))


lab.check_records_to_check(records_to_check)

# %%
print(f"{'cheat rate':>12}  {'rows to check at 95%':>22}  {'as a share of a 10M-row day':>28}")
for rate in (0.10, 0.02, 0.001, 0.0001):
    k = records_to_check(rate, 0.95)
    print(f"{rate:>11.2%}  {k:>22,}  {k / 10_000_000:>27.3%}")

# %% [markdown]
# Even a one-in-ten-thousand cheat is caught by recomputing a third of a percent of the day, so
# the costs stay modest.
#
# The weak point is the assumption above that arithmetic, which is that the operator picked the
# cheating rows *before* seeing which ones you would ask for, and that your draws are independent
# of their choice. Neither holds. The operator sees each request as it arrives, and audit traffic
# is not hard to recognise: it arrives in a burst, from one account, on prompts nobody else sends.
# A datacentre that serves the honest model to anything it suspects is being audited has a cheat
# rate of exactly zero on your sample and whatever it likes on everyone else's, and no `k` finds
# the cheat.
#
# Routing is what makes a backdoored checkpoint hard here, rather than the backdoor itself. Serving
# one backdoored model to everybody fails the digest on ordinary rows, as it did above. Serving the
# reference model to anything that looks like an audit and the backdoored one to the rest brings
# back the coverage problem, because the rows you draw are the rows they chose to serve honestly.
#
# What breaks the symmetry is who chooses the inputs. Sampling the operator's log lets them
# anticipate you; sending your own prompts, indistinguishable from real traffic, does not. Passive
# audit of a log is a much weaker instrument than active probing, and active probing costs nothing
# extra to build.

# %% [markdown]
# ## Part 4: your tolerance is their budget
#
# Everything so far let the auditor recompute on the datacentre's own machine at the datacentre's
# own precision. No auditor gets that. They arrive with their own rig: a different GPU, a
# different kernel, a different numerical precision, and the operator's fleet is not identical to
# itself either.
#
# Floating-point addition is not associative, so the same weights and the same prompt produce
# different low-order bits when the arithmetic is arranged differently. Precision is the loudest
# version of that, but batch size alone is enough: a row computed in a batch of 32 and the same
# row computed alone go through different reduction orders and come out different
# ([Thinking Machines,
# 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)).
#
# Load the auditor's rig and re-run Part 1's audit on the honest transcript, the one you already
# certified clean.

# %%
audit_model, audit_tokenizer = lab.load_audit_model()
mismatches = spot_check(transcript, lab.make_recomputer(transcript, audit_model, audit_tokenizer), 20, np.random.default_rng(0))
print(f"\nhonest transcript, auditor's own rig: {len(mismatches)} of 20 rows fail\n")
print("the equality test is dead; nobody cheated")

# %% [markdown]
# Exact comparison is gone, and with it the property that made Part 1 so cheap. The auditor has to
# fall back on a distance, meaning how far apart the claimed and recomputed logits are, and a line
# drawn across that distance. `recompute_deltas` gives the largest logit disagreement per row.

# %%
honest_deltas = lab.recompute_deltas(transcript, audit_model, audit_tokenizer)
print(f"honest disagreement: median {np.median(honest_deltas):.3f}, max {honest_deltas.max():.3f}")

# %% [markdown]
# ### Exercise: set the tolerance
#
# Pick the line. Too tight and you spend the week accusing honest operators of treaty violations
# on the strength of rounding; too loose and you certify anything. Calibrate the line the way a
# monitor is calibrated anywhere else, from a budget for how often you are willing to be wrong
# about an honest party.

# %%
@exercise
def calibrate_tolerance(honest_deltas, false_accusation_rate):
    """Smallest tolerance that wrongly accuses at most `false_accusation_rate` of honest rows.

    `honest_deltas` is an array of per-row disagreements measured on traffic you know to be honest.
    Return the smallest float `t` such that at most a `false_accusation_rate` fraction of
    `honest_deltas` is strictly greater than `t`. A rate of 0 means no honest row may exceed it.

    "At most" is the whole specification, and it has to hold on a short calibration set as well as
    a long one, which is where the obvious one-liner will let you down.
    """
    return float(np.quantile(honest_deltas, 1 - false_accusation_rate, method="higher"))


lab.check_calibrate_tolerance(calibrate_tolerance)

# %%
tolerance = calibrate_tolerance(honest_deltas, 0.05)
print(f"tolerance at a 5% false-accusation budget: {tolerance:.3f}")

# %% [markdown]
# Now the operator gets to see that number too, or estimate it, which is easier, because the noise
# floor is a property of commodity hardware rather than a secret.
#
# The cheat changes shape here. A checkpoint swap is one fixed distance from the promised model,
# and the base model sits so far outside any tolerance that it says nothing about where the line
# falls. Part 4 needs a dial, so the substitution below is gaussian noise scaled to each tensor's
# own standard deviation. It is the only synthetic substitution in the lab, and it is there because
# no real one lands close enough to the floor to show where the line is: quantise even one
# transformer block to int8 and the disagreement is several times the honest floor.

# %%
# Six fleets of 40 rows, each logged and recomputed; CI's CPU gets 10 rows a fleet.
sweep_prompts = prompts[: 10 if ci_mode() else 40]
strengths = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
sweep_deltas = {
    strength: lab.recompute_deltas(
        lab.log_run(model, tokenizer, sweep_prompts, dishonest_rows=range(len(sweep_prompts)), strength=strength),
        audit_model,
        audit_tokenizer,
    )
    for strength in strengths
}
print("measured", len(strengths), "tampered fleets")


# %% [markdown]
# ### Exercise: score the audit
#
# Report what the line actually buys. Everything above the tolerance is flagged: on cheating rows
# that is a catch, on honest rows an accusation you will have to withdraw.

# %%
@exercise
def audit_outcome(honest_deltas, cheating_deltas, tolerance):
    """Return `(caught, falsely_accused)` for flagging every row strictly above `tolerance`.

    `caught` is the fraction of `cheating_deltas` above the tolerance; `falsely_accused` is the
    fraction of `honest_deltas` above it. Both are floats in [0, 1].
    """
    caught = float(np.mean(np.asarray(cheating_deltas) > tolerance))
    falsely_accused = float(np.mean(np.asarray(honest_deltas) > tolerance))
    return caught, falsely_accused


lab.check_audit_outcome(audit_outcome)

# %%
print(f"tolerance {tolerance:.3f}, calibrated to a 5% false-accusation budget\n")
print(f"{'tamper strength':>16}  {'median delta':>13}  {'caught':>8}")
for strength, deltas in sweep_deltas.items():
    caught, _ = audit_outcome(honest_deltas, deltas, tolerance)
    print(f"{strength:>16.3f}  {np.median(deltas):>13.3f}  {caught:>7.0%}")

# %% [markdown]
# The bold tampers are caught every time. Read up the table until the catch rate falls to roughly
# the 5% you budgeted for false accusations: that row is your budget firing on schedule and
# occasionally landing on a cheating row by coincidence, not detection. Below that line a tamper's
# disagreement with your recomputation is smaller than the disagreement your own hardware produces
# on honest work, so no threshold separates the two.
#
# Where the line falls is a property of your rig rather than of the cheat. Recomputing in fp16 on a
# T4, the honest floor sits near 0.02 and only the 0.001 and 0.002 tampers disappear under it; on a
# CPU falling back to bfloat16 the floor is ten times higher and far bolder tampers hide. The
# operator does not have to guess which, because the floor is a property of commodity hardware.
#
# The tolerance you were forced into is a budget you handed the operator. Widen the tolerance to
# stop accusing honest datacentres and you widen what a dishonest one can hide; narrow it and you
# start issuing violations over arithmetic. Put the budget at 20% and at 1% and watch both ends
# move.

# %%
smallest, largest = min(strengths), max(strengths)
for budget in (0.20, 0.05, 0.01):
    t = calibrate_tolerance(honest_deltas, budget)
    caught_small, _ = audit_outcome(honest_deltas, sweep_deltas[smallest], t)
    caught_large, accused = audit_outcome(honest_deltas, sweep_deltas[largest], t)
    print(f"budget {budget:>6.1%}: tolerance {t:.3f}, accuses {accused:.0%} of honest rows, "
          f"catches {caught_small:.0%} of the {smallest} tamper and {caught_large:.0%} of the {largest}")

# %% [markdown]
# **Take it further, on your own time:**
#
# - The tamper here is undirected noise, which is the least efficient way to spend a budget. Build
#   one that spends the budget deliberately: change the model's behaviour on a narrow set of
#   prompts while moving the logits on ordinary traffic as little as possible. How much behaviour
#   can you buy under a fixed tolerance?
# - Reproduce the batch-size effect rather than the precision one: log a row alone and recompute
#   it inside a padded batch. How large is that disagreement next to the fp16/fp32 gap, and would
#   a tolerance calibrated for that gap catch anything at all?
# - The auditor never verified the *model*, only that some model reproduced the log. What does a
#   digest over the weights get you, and what does it stop getting you once the operator can load
#   one file and serve from another?
# - Serving a *smaller* model is the substitution with the strongest commercial motive and the one
#   this lab cannot demonstrate, because the digest indexes logits by token id and a model with a
#   different vocabulary has nothing to compare. Inside a family that shares a tokenizer it works:
#   claim Qwen2.5-1.5B-Instruct, serve Qwen2.5-0.5B-Instruct, and the same audit catches it. It is
#   left out here only because it means every participant downloading a 3 GB checkpoint.

# %% [markdown]
# ## What to take away
#
# Exact matching works. Twenty forward passes cleared a two-hundred-row day, and the sampling
# arithmetic puts a one-in-ten-thousand cheat at a third of a percent of the traffic recomputed.
# That covers one case: an operator who swaps a checkpoint, logs it honestly, and runs the same
# hardware you do.
#
# The lab removed each of those assumptions in turn.
#
# The operator writes the log. Recomputation compares that file against a model and never reaches
# the serving path, so a truthful log over dishonest serving passes every check. A record the
# operator cannot author means a signature from inside the chip, or an attested enclave.
#
# The operator chooses which rows to cheat on. Audit traffic is recognisable: a burst, one account,
# prompts nobody else sends. Sampling their log is sampling what they chose to show you. Choosing
# your own prompts avoids that and needs no new infrastructure.
#
# The auditor's arithmetic differs from the operator's. Equality becomes a tolerance, and the
# tolerance covers any change smaller than what your own hardware does to honest work. On a T4 in
# fp16 that was a weight change of 0.002 standard deviations. Reproducible kernels are a
# prerequisite for this mechanism.
#
# Day 4's treaty-verification session covers mechanisms with the same three weak points: who
# authors the evidence, who selects the sample, and whose hardware is the reference.
# %%
# @lab-only
# Stuck on spot_check? rng.choice(len(records), size=k, replace=False) draws the rows. Call
# recompute(i) once per drawn row and keep the ones where it disagrees with records[i].digest.
#
# Stuck on records_to_check? k draws all miss with probability (1 - cheat_rate) ** k. Solve
# 1 - (1 - cheat_rate) ** k >= confidence for k and round up, with math.ceil and math.log.
print("rng.choice for the sample; solve (1 - rate) ** k for the sample size")
