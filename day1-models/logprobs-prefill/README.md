# Logprobs, output distributions, instruction hierarchies & prefill

**Day 1 · 1.5–2h · Lab · GPU optional (a 0.5B model; runs on CPU in a couple of minutes)**

## Objective

Stop reading a model's output as a string it chose to say, and start reading it as a
distribution you can measure and steer. "Helpfulness", "refusal", "following instructions" —
each is a shape in that distribution plus a decision about how to sample from it. By the end,
a refusal is not a property the model has but a probability someone assembling the prompt can
move.

## What it surfaces

- **A model is a sampling policy, not a fixed responder.** Temperature, top-p and entropy make
  "the model is confident" a number, and make two deployments of identical weights visibly
  different systems.
- **Scoring a completion the model didn't write** — the one technique the rest of the week
  rests on. It's how evals grade multiple-choice options and how the later parts measure
  behaviour without generating it.
- **The instruction hierarchy is a learned preference, not a boundary.** Participants put the
  same instruction in the system prompt, the user turn, and a retrieved document, and read off
  how much authority each channel actually carries. This is the mechanism behind indirect
  prompt injection, and Day 2 picks it up as a control problem.
- **Prefill.** Writing the first tokens of the assistant's own turn makes its own refusal
  collapse — measured, then watched in generation. Which deployment surfaces expose this
  (open weights: always; most chat APIs: no) is the defensive takeaway, and it sets up Day 3,
  where the manipulation moves from the prompt into the weights.

## Structure

Four parts, each building on the last: read the distribution → score a completion → locate an
instruction's authority → prefill. Every function you implement is checked in the notebook
(`lab.check_*`), so you know it's right before moving on.

## Prerequisites

Day 0 done. Comfort with logits and softmax; no prior work with chat templates or the
transformers generate API assumed — the notebook builds those up.

## A note on the Day 3 preview

Part 4 uses a single, clinical dual-use prompt to show a refusal breaking under prefill. It's
generated at runtime and never committed as content; the solution notebook's committed output
is truncated to the point where the behaviour flips, not the payload. This is the offensive /
defensive pairing the program is built on — the lesson is that a refusal measured through a
chat UI does not survive a deployment where someone else controls the prompt string.
