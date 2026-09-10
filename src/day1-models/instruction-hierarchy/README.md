# Instruction hierarchies & prefill

**Day 1 · 1h30 · Lab · GPU optional (a 0.5B model runs on CPU in a couple of minutes)**

## Objective

Treat instruction-following and refusal as properties of the prompt string rather than fixed
traits of the model. You measure how much authority each input channel really carries, break the
ranking two ways, and build the input-side half of a defence.

## What it surfaces

The instruction hierarchy (system over user over retrieved content) is a learned preference, not
a wall the architecture enforces. You measure each channel's authority and read the ratio between
them, which is the mechanism behind indirect prompt injection.

Prefill is the second break. Writing the first tokens of the assistant's own turn collapses its
refusal, measured first and then watched in generation. Which deployment surfaces expose this
(open weights always, most chat APIs not) is the security point, and it sets up Day 3.

The defensive exercise is a cue-based injection scanner. It shows the cheap input-side control and
its limits at the same time: easy to write, easy to evade, worth doing, and not a boundary. That
gap is what the control agenda on Day 2 exists to close.

## Structure

Three parts: locate an instruction's authority across channels, prefill a refusal into collapse,
then scan retrieved content for injected instructions. Each function you write is checked in the
notebook.

## Prerequisites

Day 0 done. This lab hands you the scoring function from **Output distributions & scoring**, so
you can take it without having done that lab first.

## A note on the Day 3 preview

Part 2 uses one clinical dual-use prompt to show a refusal breaking under prefill. It's generated
at runtime and never committed as content, and the committed solution output stops at the point
the behaviour flips rather than the payload. Offense sits next to its defensive reading
throughout, which is how the program treats this material.
