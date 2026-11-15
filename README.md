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

## Flight data handoff (BVLM-6)

This branch prepares the published flight data and renders the original flight
text as OCR image cards. A model run can compare the same options as direct
text or readable images. The visual model run and its comparison with the
text baseline remain open.

### Start locally (Python 3.9+, no extra packages)

Source: [Zenodo 17677329, data.zip](https://zenodo.org/records/17677329), MD5
`a341ca74418ed9152a31cc3f91e261dd`. From a fresh clone, download once:

```bash
mkdir -p data/raw
curl --fail --location --output data/raw/zenodo-17677329.zip \
  'https://zenodo.org/records/17677329/files/data.zip?download=1'
python3 run_flights.py
```

An existing local ZIP needs no download. The first preview extracts only the
original README and two evaluation files to `data/raw/zenodo-17677329/data/`.
Every normal load checks the archive checksum and refuses to replace changed
originals. Data stays ignored by Git. In VS Code, open `run_flights.py` and select
**Run Python File in Terminal**. Default: user 0, test set 1, no observations.

```bash
python3 run_flights.py --set 1 --answers
python3 run_flights.py --user 12 --set 7 --n-obs 3
python3 run_flights.py --n-obs 2 --export data/generated/user0-nobs2.jsonl
python3 prepare_training.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

For user 0 / set 1 the inspection view shows utilities `[-2.3, -1.2, -2.9]`
and stored choice **Flight 2 (`rec=1`)**. `--answers` is local inspection only;
exports always exclude current solutions and private weights. The same default
appears each run because it selects the same published case; use `--set` to
browse another. There are 100 heldout sets of three flights shared by 624 users,
plus five observed three-flight rounds per user. Repeated flights are allowed.

### Handoff to models and image cards

Import from `src` (`PYTHONPATH=src`):

- `PaperDataset.case(user, n_obs, set_index)` in `bvlm.flight_data` returns
  `{source, user, n_obs, set, options, card_text, history}`. `options` preserves
  the exact paper strings and option order. `history` contains only the first
  `n_obs` observed rounds, each with `options` and observed `user_idx`.
- `render_flight_card(options)` in `bvlm.flight_cards` produces our readable
  four-feature layout. This is custom formatting, not the original model
  prompt. For a faithful paper reproduction keep the original strings and
  use the paper's prompt protocol. For a text/image comparison render exactly
  the same options/history, never sort them or include current solutions.
- `--export` writes 100 JSONL input cases for one user/observation count. The
  model runner separately produces `{"user":0,"n_obs":2,"set":1,"rec":1}`.
  Team ranges: user 0–623, n_obs 0–5, set 0–99, rec 0–2; Flight 1/2/3 maps to
  rec 0/1/2. All 374,400 combinations belong to the full evaluation run.
- `training_card(record, prefix_row, round_index)` in `bvlm.training_data`
  gives `{prefix_row, profile_idx, round, options, card_text}` for an actual
  training offer. Enumerate the prepared JSONL to obtain `prefix_row` (0-based).
  Use `profile_idx` only as source metadata: `idx` repeats and is not a unique
  dialogue ID. Do not replace training offers with evaluation images.

### Generate OCR benchmark cards

Open `generate_ocr_cards.py` in VS Code and click **Run Python File in
Terminal**. With no arguments, it uses `data/generated/ocr/` for the full set
of 100 shared held-out cards and 3,120 interaction cards. If the matching
manifest and all PNGs already exist, it prints where to find them.

For a fresh setup, install Pillow once into the project's Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ocr.txt
.venv/bin/python generate_ocr_cards.py
```

Browse `heldout/set-001.png` for a test offer or `interaction/user-000/`
for all five observed offers for user 0. Each PNG shows three flight options.
Use `--out` to choose a fresh directory if you need to regenerate the set.

Each `manifest.jsonl` row maps a PNG to its exact original option strings and
IDs (`set` for shared test cards, `user` and zero-based `round` for interaction
cards). The images preserve the baseline's wording and option order. Attach
feedback to the dialogue separately. The cards and manifest contain only the
options. The PNGs use the same four-feature data as
[the cluster text benchmark](benchmarks/text/README.md).

Git tracks the generator and its requirements. The local Zenodo archive, PNGs
and manifest are ignored. Run the generator once on each checkout that needs
the images. For an image evaluation, use the manifest to select each PNG, send
the PNG and question to an image-capable model, and write the same prediction
fields as the text benchmark. Keep the `options` strings for the text condition.

### Training data and validation

`prepare_training.py` validates and copies the two official teacher files to
`data/generated/training/{oracle,bayesian}.jsonl` without changing any bytes.
Each has **31,200 prefixes of 6,240 five-round dialogues**, with prefixes of
one through five rounds. The first two rounds are printed for inspection.
Oracle and Bayesian recommendations can differ; matched offers and subsequent
actual user choices agree. Roles, histories and teacher answers are preserved.
The final answer has no subsequent feedback in the prefix; its provenance is
the verified archive. No new teacher labels are inferred by this preparation.

The later trainer consumes these message lists in two separate runs from the
same model checkpoint. Tokenization, assistant-target masking, model setup and
GPU training belong to that trainer. This branch does not train the Bayesian
reference: its published recommendations are already in the source data.

Tests check feature conversions, option order, every stored evaluation choice,
input/answer separation, paired training prefixes and exact source copies.
Original-data tests require the ZIP; without it they are explicitly skipped.
The included four-feature evaluation artifact uses seed 0, whereas the paper
reports multiple seeds. One heldout set also occurs in user 0's first observed
round; we preserve this source fact. Preparing this archive alone is therefore
not a complete reproduction of every paper result.
