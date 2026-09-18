# Insecure-code control hackathon

**Day 2 · Hackathon · CPU only (the models run on the referee, not your runtime)**

## Objective

Build both halves of a monitoring setup and play them against each other on a live board. As blue,
build the smallest monitor that still catches insecure code an untrusted model tries to slip past
you. As red, build a model organism — a prompt, not a trained model — that writes code which does
the task but hides a real vulnerability well enough to fool the room's monitors. Swap sides partway
through: attack the assumptions you just defended.

## What it surfaces

Monitoring is the workhorse of AI control, and it has two knobs the abstract version hides: how
*small* a model you can catch with, and how *quietly* a capable model can slip past. This exercise
makes both concrete. Blue teams start with a big open-weight model and discover how far a sharper
prompt lets them walk *down* to a tiny one before the catch-rate collapses — the same trade a real
deployment makes when it can't afford to run its best model on every action. Red teams learn that the winning
attack isn't the blatant `os.system(user_input)` the house monitors already catch; it's the insecure
code that reads as ordinary, and that a *strong* judge still calls a vulnerability. The two scores
are in tension across the room: every monitor that gets cheaper invites a subtler attack, and every
attack that lands forces the monitors to re-prove themselves.

The security content is real, not a metaphor. The tasks are drawn from
[SecurityEval](https://github.com/s2e-lab/SecurityEval): small Python functions with a
security-sensitive operation, played on a handful of high-signal CWEs — command injection, SQL
injection, path traversal, unsafe deserialization, weak cryptography.

## Structure

One notebook, one live board. The notebook opens with the task suite and the weaknesses in scope,
then two short warm-ups that build the scoring you're judged on (reading a monitor's caught- and
false-flag-rates; counting the monitors an attack evades). The open competition is the rest of the
session: pick a model and write a monitor prompt, dry-run it against a calibration set, submit; then
switch to red, write an attack prompt, dry-run to see the judge's verdict, submit. The board re-scores
the whole field live.

Ground truth is a strong **judge** model, independent of both teams: a red snippet counts only if the
judge agrees it both does the task and is genuinely insecure. The referee never runs any submitted
code — it is only ever read, by the judge and by the monitors — so the exercise has no
code-execution risk to manage.

## Prerequisites

Day 0 done. No GPU. Comfort reading short Python and recognising an insecure pattern when named (the
notebook lists the CWEs in scope). The assume-breach framing from earlier in the day is the mindset
this rewards: treat the capable model as possibly hostile and design the monitor for its worst
behaviour, not its typical one.

## Reading

The control agenda and what a monitor is for: [Greenblatt et al., 2023](https://arxiv.org/abs/2312.06942).
Model organisms — building a specific misbehaviour on purpose so you can study and detect it — from
[Hubinger et al., *Sleeper Agents*, 2024](https://arxiv.org/abs/2401.05566). The task suite:
[Siddiq & Santos, *SecurityEval*, 2022](https://doi.org/10.1145/3549035.3561184).
