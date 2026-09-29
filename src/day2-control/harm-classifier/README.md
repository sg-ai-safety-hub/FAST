# Train a harm classifier, then break it

**Day 2 · 60–90 min · Lab · T4 GPU (~6 min of compute, mostly one fine-tune)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sg-ai-safety-hub/FAST/blob/main/src/day2-control/harm-classifier/lab.ipynb)

> **Content warning:** the lab uses real prompts from a safety dataset, including violence, self-harm,
> sexual content and hate speech. Only short excerpts are printed.

## Objective

Train the cheapest monitor there is, a classifier that flags harmful prompts. Set its threshold the
way control does, from a false-flag budget rather than accuracy. Then attack it, patch one hole, and
see what the patch doesn't cover.

## What it surfaces

In production the first filter on every request is usually a small trained classifier, not a
prompted LLM, so its holes are the ones an attacker meets first. Some are distributional:
leetspeak, encodings, other languages. Padding is different. It beats a truncating model because the
request is never read, and a non-truncating one because benign filler dilutes it, and windowed
scoring fixes only the first. Retraining on an evasion closes that evasion and its near neighbours,
not the next family of attacks. The labels are noisy too: some of the model's most confident
"errors" are harmful prompts labelled benign.

## Structure

Train a TF-IDF baseline and a fine-tuned DistilBERT, with the training code in the lab package. Set
thresholds at a 5% false-flag budget and compare the two slice by slice. Then a play box, four
attacks, a windowed-scoring fix, a patch-and-retrain step, and a look at the most confident errors.
Three functions you write are checked in the notebook: the threshold, the evasion rate, and
windowed scoring.

The ML is handed to you; the exercises are about thresholds and attacks. For the ML half of the
room, the part to dwell on is the threshold. A monitor is deployed at a false-flag budget, not at
its best F1.

## Prerequisites

Day 0 done. Best run before the control hackathon, which scores monitors on the same trade-off.

## Data

[Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) (NVIDIA,
CC-BY-4.0), public and ungated. It's downloaded at runtime and never vendored, then curated to
prompts and labels, with redacted, contradictory, duplicate and train/test-leaked rows removed.
Ghosh et al., *Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM
Guardrails*, NAACL 2025, [arXiv:2501.09004](https://arxiv.org/abs/2501.09004).
