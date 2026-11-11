# Bayesian VLMs: Teaching Vision-Language Models to Weigh Visual Evidence

**Multimodal Artificial Intelligence Project Lab | TU Darmstadt | Winter Semester 2026/27**

## Overview

This research project investigates whether Bayesian teaching can improve belief updating in Vision-Language Models (VLMs) and how well these improvements generalize to unseen visual tasks.

Building on previous research in Bayesian belief teaching for Large Language Models (LLMs), we aim to extend these methods to multimodal settings.

## Research Questions

**Main Research Question:**

Can Bayesian teaching improve belief updating in VLMs, and how well does it generalize?

**Additional Research Questions:**

- Is Bayesian teaching more effective than oracle teaching in a visual setting?
- Does reasoning improve belief updating before fine-tuning?

## Methodology

Our research is divided into three phases:

### 1. Research & Familiarization

- Review related literature on Bayesian teaching and probabilistic reasoning.
- Select suitable vision-language models.
- Define evaluation metrics.

### 2. Dataset Generation & Baseline Evaluation

- Generate visual tasks with known probability distributions.
- Reproduce and extend existing benchmarks.
- Compare reasoning and non-reasoning models.
- Evaluate model predictions against exact Bayesian references.

### 3. Fine-Tuning & Evaluation

- Fine-tune VLMs using Bayesian belief traces.
- Compare fine-tuned models against baseline performance.
- Evaluate generalization to previously unseen tasks.

## Evaluation Metrics

- **Success Rate:** Measures task completion performance.
- **Agreement:** Measures agreement with the Bayesian Assistant's decisions.
- **KL Divergence:** Measures differences between predicted and exact Bayesian probability distributions.

## Expected Contributions

- A visual belief-updating benchmark with known Bayesian reference distributions.
- An empirical evaluation of Bayesian-taught VLMs.
- Insights into the generalization of Bayesian belief updating across visual tasks.

## Team

**Project Group 1 – TU Darmstadt**

- Christopher Tutsch
- Eren Çivi
- Julius Pantel
- Jens Thomae
- Ankita Upadhyay

**Supervisor:** Tobias Wieczorek

## References

1. [Bayesian teaching enables probabilistic reasoning in large language models](https://www.nature.com/articles/s41467-025-67998-6) — Nature Communications.
2. [Apple Research – Related Work](https://arxiv.org/abs/2605.06915v1)
3. [Bayesian Fine-Tuning – Follow-up Research](https://arxiv.org/abs/2610.00679)
4. [Probabilistic Reasoning and Decomposition](https://arxiv.org/abs/2609.38005)
5. [Bayesianity in Language Models](https://arxiv.org/abs/2406.00793)

## Project Status

Research and initial benchmark development are in progress.

Additional documentation, datasets, experimental results, and implementation details will be added throughout the project.

## License

No open-source license has been selected at this stage.