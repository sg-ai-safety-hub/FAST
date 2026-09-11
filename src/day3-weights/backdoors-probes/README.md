# Backdoors and probes

**Day 3 · 75 min · Lab · GPU not needed (the model is tiny)**

## Objective

Plant a backdoor in a model by poisoning its training data, confirm that ordinary behavioural
testing can't see it, then catch it anyway with a probe that reads the model's activations. By the
end you've built both halves: the attack that hides a trigger-conditioned behaviour, and the
detector that finds evidence of it from the inside.

## What it surfaces

A backdoor is installed through data, not weights: poison a small fraction of the training set so
a trigger forces a target output, and the model learns the real task and the trigger behaviour
together. Behavioural evaluation misses it because the trigger is one point in a space you can't
enumerate, so "we tested it and it behaved" is a weak claim for a model whose training you didn't
control.

The trigger still leaves a signature in the activations, and a difference-in-means probe reads it
off well above chance while behaviour on clean inputs stays perfect. That's the defensive shape:
when you can't trust behaviour, read the internals. It's the same construction as the abliteration
lab, pointed at a hidden presence instead of a removed one.

## Structure

Three parts: poison the data and train the backdoor, measure that it's invisible to behavioural
testing while the trigger works, then build the activation probe that catches it. The model is a
small classifier you train from scratch in seconds, standing in for a language model so the whole
loop fits in one sitting. Each function you write is checked in the notebook.

## Prerequisites

Day 0 done. No training-loop experience assumed; the loop is provided and you write the
data poisoning, the metrics, and the probe.

## Reading

The full-scale version of this is the sleeper-agents work
([Hubinger et al., 2024](https://arxiv.org/abs/2401.05566)) and the finding that linear probes
catch it ([Anthropic, 2024](https://www.anthropic.com/research/probes-catch-sleeper-agents)).

## A note on dual-use

The backdoor is a label flip on a toy classifier, not a code-vulnerability model, and nothing
trained here is redistributable. The attack ships with its detector in the same notebook, which is
how the program treats this material.
