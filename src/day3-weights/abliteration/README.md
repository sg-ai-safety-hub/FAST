# Abliteration

**Day 3 · 75 min · Lab · GPU recommended (a 0.5B model fits a T4)**

## Objective

Remove a safety-tuned model's refusal behaviour by finding the single direction in its residual
stream that carries refusal and projecting it out, first live during generation and then baked
into the weights. By the end you can turn a checkpoint that refuses into one that doesn't, with a
few dozen prompts and one pass over the parameters.

## What it surfaces

Refusal is mediated by roughly one direction, so it can be located by a difference in means and
removed by a projection. Doing it as a runtime hook is reversible and needs live access; doing it
to the weights is permanent and ships in the file.

The security point is how cheap the permanent version is. Refusal training shifts a model's
default behaviour, which is worth something when the provider holds the weights and you only send
text. It does almost nothing against anyone who can read and edit the parameters, and that gap
doesn't close once the file is released. This is the concrete reason weight release is a
one-way decision.

## Structure

Four parts: find the refusal direction from paired prompts, project it out of the activations as
the model generates, orthogonalise the weights so the change is permanent, then work through what
that means for weight security. Each function you write is checked in the notebook. A fifth
optional part measures the capability tax the edit costs, for anyone who finishes early or wants to
come back to it.

## Prerequisites

Day 0 done. The maths is the difference in means and projection from the worked template, now on
a real model. Comfort with a model's residual stream helps but the notebook builds up what it
needs.

## A note on dual-use

The lab runs on a small open model with a short, clinical set of prompts generated at runtime,
and nothing weaponizable is committed. The offensive technique and its defensive reading (what to
do about it, in Part 4) sit together, which is how the program treats this material.
