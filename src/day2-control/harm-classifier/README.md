# Train a harm classifier, then break it

**Day 2 · 1h30 · Lab · T4 or L4 GPU (a few minutes of compute, most of it one fine-tune)**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sg-ai-safety-hub/FAST/blob/main/src/day2-control/harm-classifier/lab.ipynb)

> **Content warning.** The lab trains on real prompts from a safety dataset, including requests
> involving violence, self-harm, sexual content and hate speech. The notebook shows only short
> truncated excerpts.

## Objective

Build the cheapest monitor there is, a small classifier that flags harmful prompts, and score it the
way control scores any monitor: catch rate at a fixed false-flag budget. Then attack it, find where
it fails, patch one failure, and measure what the patch cost.

## What it surfaces

The hackathon's monitors are prompted LLMs. In practice the first line in front of every request is
a trained classifier, because it is cheap enough to run on all traffic. That makes its failure modes
the ones an attacker meets first. Some holes are distributional: typos, leetspeak, other languages,
encodings, anything unlike the training data. One is architectural: the model reads 256 tokens, so a
request padded past that is never seen, and that gets an engineering fix (score in windows) rather than
a data fix. Every fix moves both numbers. Windowing raises false flags on long inputs, and training
on one evasion closes that evasion and not the next. The labels themselves are only moderately
agreed on by humans, so some "errors" are arguable.

## Structure

Explore the data, then set a baseline (TF-IDF + logistic regression) with its threshold chosen at a
5% false-flag budget. Fine-tune DistilBERT and compare the two slice by slice (short vs long prompts,
borderline labels, harm category). A play box scores anything you type. Hole hunting measures evasion rates for
perturbations and padding, fixes truncation, probes over-flagging, and reads the most confident
errors against the borderline labels. It ends with a patch-and-retrain step. Three functions you
write are checked in the notebook: the threshold, the evasion rate, and windowed scoring.

**For the security half of the room:** the ML is handed to you. The training loop is about twenty
lines to read, not write, and the exercises are about thresholds and attacks. **For the ML half:**
the part to dwell on is the threshold. Accuracy and F1 aren't how a monitor gets deployed, and a
false-flag budget is.

## Prerequisites

Day 0 done. Useful before the control hackathon, since it sets up the catch-rate / false-flag
trade-off the hackathon scores. The Day 1 instruction-hierarchy scanner is the hand-written
counterpart, though not required.

## Data

[Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) from NVIDIA,
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), public and ungated. The notebook downloads
it from Hugging Face at runtime (no login) and curates it in a few lines
(`fast.labs.day2_control.harm_classifier.curate`): prompts and labels only, redacted and
contradictory rows dropped, duplicates and train/test leakage removed. Nothing is vendored here.
Ghosh et al., *Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM
Guardrails*, NAACL 2025, [arXiv:2501.09004](https://arxiv.org/abs/2501.09004).
