# Train a harm classifier, then break it

**Day 2 · 60–90 min · Lab · T4 GPU (~6 min of compute, mostly one fine-tune)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sg-ai-safety-hub/FAST/blob/main/src/day2-control/harm-classifier/lab.ipynb)

> **Content warning:** the lab uses real prompts from a safety dataset, including violence, self-harm,
> sexual content and hate speech. Only short excerpts are printed.

## Objective

Train the cheapest monitor there is, a classifier that flags harmful prompts. Set its threshold the
way control does, from a false-flag budget rather than accuracy. Then red-team it with your own
prompts and attacks.

## What it surfaces

In production the first filter on every request is usually a small trained classifier, not a
prompted LLM, so its holes are the ones an attacker meets first. Some are distributional:
leetspeak, encodings, other languages. Padding is different. It beats a truncating model because the
request is never read, and a non-truncating one because benign filler dilutes it.

## Structure

Train a TF-IDF baseline and a fine-tuned DistilBERT, with the training code in the lab package. Set
thresholds at a 5% false-flag budget and compare the two slice by slice. Red-team both, first by
hand in a play box, then with an attack you write and score against four built-in ones. Two
functions you write are checked in the notebook: the threshold and the evasion rate.

The ML is handed to you; the exercises are about thresholds and attacks. Security engineers get a
concrete feel for how a trained filter sets its threshold and fails under distribution shift. ML
engineers get the adaptive attacker: a model that scores well on test and still loses to the
first recipe someone writes against it.

## Prerequisites

Day 0 done. Best run before the control hackathon, which scores monitors on the same trade-off.

## Data

[Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) (NVIDIA,
CC-BY-4.0), public and ungated. It's downloaded at runtime and never vendored, then curated to
prompts and labels, with redacted, contradictory, duplicate and train/test-leaked rows removed.
Ghosh et al., *Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM
Guardrails*, NAACL 2025, [arXiv:2501.09004](https://arxiv.org/abs/2501.09004).
