# Live control arena

**Day 2 · Hackathon (second half) · real models via a hosted referee**

## Objective

Play the control game from the first half again — a protocol wrapping an untrusted model, scored on
safety and usefulness — but with a real model as U and real coding tasks underneath. Teams write the
same two things they wrote against the synthetic arena, now as a **function plus a prompt**, and
compete on a live board.

## What it surfaces

The synthetic arena let you reason about protocols fast because you could see the machinery. This
half takes the machinery away: there's a real model on the other side, and you can't reduce it to a
formula. The interface is deliberately identical, so the *only* thing that changes is that the
signals are real — and that's the lesson.

The sharpest version of it is on the red side. In the synthetic arena, `subtlety` was a dial: you
set it to 0.7 and your backdoor was, by definition, that quiet. Here subtlety is **not something you
set** — it's whatever the referee measures from what your prompt actually made the model produce:
how suspicious the monitor finds the code, and whether the backdoor even fires. The gap between the
attack you intended and the attack your prompt bought you is the whole exercise. Blue feels the
mirror image: a monitor prompt is only as good as its ability to separate real backdoored code from
real clean code without crying wolf, and the protocol's thresholds mean nothing if the prompt
behind them is vague. It connects straight to the Day 1 instruction-hierarchy work: this is
prompt-level attack and defence, wrapped in a control protocol.

## Structure

A single notebook (`arena.ipynb`). Blue writes a `protocol(obs)` — the accept / defer / audit / edit
decision from the first half — and a `MONITOR_PROMPT`. Red writes an `attack(view)` returning how
often to arm a backdoor per task, and an `ATTACK_PROMPT`. You dry-run a backdoor prompt to see
whether it passes the tests and fires, then submit; you iterate against what the board shows.

All scoring runs on a hosted referee (`server/`), which drives the models, grades the generated code
in a sandbox, and runs the blue-versus-red tournament on real coding tasks (APPS-style). The board
ranks blue by worst-case safety across the current attacks and red by how far it drives the best
protocol down, and shows the current title fight. There's no separate solution notebook — the answer
key here is the board.

This is the second half of the Day 2 hackathon; the first half is the synthetic
[`control-hackathon`](../control-hackathon/), which builds the vocabulary this half assumes.

## Prerequisites

The synthetic control hackathon first — it establishes protocols, attacks, and the
safety/usefulness trade, and this half reuses its exact interface. No GPU; everything runs through
the referee, so you also need the arena link your instructor shares.

## Dual-use note

Writing backdoor-insertion prompts is genuinely dual-use. It's taught here because the defensive
half — the monitor that catches it — is the same exercise, and understanding the attack is how you
counter it. It stays on toy coding tasks and small models, and nothing it produces is kept.

## Reading

The control setting is [Greenblatt et al., 2023](https://arxiv.org/abs/2312.06942). The live version
here reproduces the *shape* of AISI + Redwood's
[ControlArena](https://github.com/UKGovernmentBEIS/control-arena) — a protocol and an attack policy
scored over a task set with an audit budget — on a thin harness rather than running the framework
itself; untrusted monitoring and collusion, the natural next escalation, are discussed in the paper.
