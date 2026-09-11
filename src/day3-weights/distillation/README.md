# Distillation

**Day 3 · 75 min · Lab · GPU recommended (a 0.5B teacher and a small student fit a T4)**

## Objective

Train a small student model to reproduce a larger teacher, two ways: white-box from the teacher's
full output distribution, and black-box from its generated text alone. By the end you can state
what an attacker actually needs to clone a model, and why more access makes it cheaper.

## What it surfaces

Capability transfers without copying weights. A random-init student trained on a teacher's outputs
ends up echoing it, so reproducing a model doesn't require exfiltrating it.

The two paths aren't equal, and the gap is the security-relevant part. White-box distillation reads
the teacher's whole distribution, including its relative confidence across tokens it didn't pick,
so it learns fast. Black-box gets one token per position and needs far more data. The more of the
distribution a provider exposes, through raw logits or top-k probabilities, the closer an API sits
to handing over the weights. This reframes the weight-security perimeter: a model's own responses
are a copying channel too.

## Structure

Three parts: distill white-box with a temperature-softened KL loss, distill black-box with ordinary
cross-entropy on the teacher's text, then compare the fidelity of the two and what it costs. A full
distillation run is longer than a session, so the runs here are short — enough to watch the loss
fall and the student begin to move toward the teacher. Each function you write is checked in the
notebook.

## Prerequisites

Day 0 done. Comfort with softmax, logits, and cross-entropy. The **Output distributions & scoring**
lab from Day 1 covers the distribution mechanics this builds on, though it isn't required.

## Reading

The soft-target method is [Hinton et al., 2015](https://arxiv.org/abs/1503.02531). The
OpenAI/DeepSeek dispute is the live case of black-box distillation at scale, and why terms of
service now forbid training on model outputs.
