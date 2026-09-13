# AI control hackathon

**Day 2 · Hackathon · CPU only (the arena runs in seconds, no GPU)**

## Objective

Build both halves of a control setup and play them against each other. As blue, design a protocol
that gets useful work from an untrusted model without letting a scheming one slip a backdoor past
you. As red, play the untrusted model and find the attack that beats it. Everything lands on a
shared leaderboard as a safety-versus-usefulness point, so the room can see the whole frontier at
once.

## What it surfaces

Control is a game, and the score is two numbers in tension. Deferring to a weaker trusted model is
always safe and always costs you capability; trusting the strong model is useful and open to
attack; a few well-spent human audits buy back most of the safety without much of the cost. Playing
both sides makes that tradeoff concrete: blue teams feel the audit budget bite, and red teams learn
that the winning attack is the rare, subtle one that lands just below the audit threshold, not the
blatant one. Switching roles halfway is the point — you attack the assumptions you just relied on.

## Structure

The notebook opens with the arena and a house frontier to beat, then two short warm-up exercises
(the monitoring decision, and the defer-to-resample decision) that give you the primitives. The
open competition is in between: tune a blue protocol, switch to a red attack, and submit each to
the live leaderboard. Blue protocols pick among four actions — accept U, defer to T, audit, or let
T edit the solution — and can pace a scarce audit budget across the stream; red picks where and how
subtly to strike. A defer-to-resample interlude sits in the middle as a guided detour: guess the
safety-usefulness curve, then compute it.

The board is a small hosted referee (`server/`, a container you deploy to Cloud Run). It keeps
every submission, continuously re-runs the whole blue-versus-red tournament, and shows the live
standings, so a strong attack can knock the leading protocol off the top in real time. The game
runs on a table of synthetic signals and never calls a model, which keeps that round-robin cheap.
The back half of the afternoon plays the same game against a real model — that's a separate
exercise and a separate referee, [`live-control`](../live-control/), where the protocol and attack
you tuned here come back paired with the prompts that drive the real models.

## Prerequisites

Day 0 done. No GPU. Comfort reading a threshold policy in a few lines of Python. The assume-breach
framing from earlier in the day is the mindset this rewards: treat the capable component as
possibly hostile and design for its worst behaviour, not its typical one.

## Reading

The setting and the protocols are from [Greenblatt et al., 2023](https://arxiv.org/abs/2312.06942).
Defer-to-resample is the resampling idea from the follow-up work on controlling agents
([Ctrl-Z, 2025](https://arxiv.org/abs/2504.10374)).
