# Output distributions & scoring

**Day 1 · 1h30 · Lab · GPU optional (a 0.5B model runs on CPU in a couple of minutes)**

## Objective

Read a model as what it is: a probability distribution over the next token that something samples
from. By the end you can shape how it's sampled, measure how sure it is, and score any string
against it, and you've used that scoring to build a small multiple-choice evaluator.

## What it surfaces

A "deterministic" model is a sampling policy. Temperature and top-p make two deployments of one
checkpoint behave differently, and that setting rarely shows up in a model card.

Entropy turns "the model is confident" into a number, which is how you tell reciting apart from
guessing.

Scoring a string the model didn't generate is the technique the rest of the week rests on. An
eval is that scoring in a loop: score each choice, take the argmax, count how often it's right.
The accuracy printed in a system card is this exact procedure, and once you've built it you can
see where it misleads, from choice phrasing to length bias to which tokens get counted.

## Structure

Three parts: read the distribution (softmax, temperature, entropy, nucleus sampling), score a
completion, then assemble the evaluator. Every function you write has a check in the notebook, so
you know it's right before moving on.

## Prerequisites

Day 0 done. Comfort with logits and softmax. No prior work with the transformers API assumed; the
notebook builds it up.

## Pairs with

Runs back to back with **Instruction hierarchies & prefill**, with a break between. The two are
independent: the scoring function you build here is handed to you there.
