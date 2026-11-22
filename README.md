# Bayesian VLMs: Teaching Vision-Language Models to Weigh Visual Evidence

**Multimodal Artificial Intelligence Project Lab | TU Darmstadt | Winter Semester 2026/27**

To build the flight benchmark PNGs, see the [OCR card setup](#generate-ocr-benchmark-cards)
and [diagram cards](#generate-diagram-cards-and-cards-with-noise-bvlm-78).

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

For the text preview and teaching dialogue export, download the original
[Zenodo archive](https://zenodo.org/records/17677329) once. Its MD5 is
`a341ca74418ed9152a31cc3f91e261dd`.

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

The generator reads the original flight JSON files already included under
[the text benchmark's data directory](benchmarks/text/data/eval). You can
generate all 3,220 PNGs from a fresh checkout without the Zenodo download.

From the repository root, create a Python environment, install Pillow, and
run the generator:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ocr.txt
.venv/bin/python generate_ocr_cards.py
```

In VS Code, you can also select the environment with Pillow, open
`generate_ocr_cards.py`, and click **Run Python File in Terminal**.

Find the output under `data/generated/ocr/`. It contains 100 shared test
cards in `heldout/` and 3,120 earlier offers in `interaction/`. Each PNG
shows three flight options. If the matching manifest and all PNGs already
exist, the script prints where to find them. To regenerate the cards in a
new directory, pass `--out`.

Each `manifest.jsonl` row maps a PNG to its exact original option strings and
IDs (`set` for shared test cards, `user` and zero-based `round` for interaction
cards). The images preserve the baseline's wording and option order. Attach
feedback to the dialogue separately. The cards and manifest contain only the
options. The PNGs use the same four-feature data as
[the cluster text benchmark](benchmarks/text/README.md).

Run the generator once in each working copy where you need the images. For an
image evaluation, use the manifest to select each PNG, send
the PNG and question to an image-capable model, and write the same prediction
fields as the text benchmark. Keep the `options` strings for the text condition.

### Generate diagram cards and cards with noise (BVLM-7/8)

Install Pillow with the [OCR setup above](#generate-ocr-benchmark-cards).
From the repository root, run:

```bash
.venv/bin/python generate_ocr_cards.py
.venv/bin/python generate_diagram_cards.py
```

You can also open either script in VS Code and click **Run Python File in
Terminal**. The scripts use the flight JSON already included in the repo.
Each output directory contains 3,220 cards, arranged as 100 held-out offers
and 3,120 earlier interaction offers.

- `data/generated/ocr/` contains the original flight descriptions as images.
- `data/generated/diagrams/` shows the same three flights using four charts.
  Departure time and stops are plotted as dots, duration and price as bars.
  The axes are fixed across cards.
- `data/generated/diagrams-noise/` adds two unrelated notices and one unrelated
  chart. The flight charts have the same pixels and positions as the clean
  cards. The card path seeds the selection of distractors.

All three formats use the same image paths and manifest rows, including
`heldout/set-001.png` and `interaction/user-000/round-0.png`. The manifest
preserves the three original flight descriptions per offer. To generate
diagram cards in a new parent directory, pass `--out` to the diagram generator.
The two diagram sets will be written into that directory.

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

### Ada four-format benchmark (BVLM-14)

The Ada runner evaluates the official BF16 Qwen3.5-9B checkpoint with vLLM.
It follows the interactive text reference at commit
[9128b3f](https://github.com/c-tutsch/Bayesian-VLMs/tree/9128b3fc4ab31ac16972010eed8eff654d3c2fb7/benchmarks/text).
For each format, the model recommends a flight in each history round and receives
the same stored user choice as feedback. Its own previous answers remain in
the conversation. Held-out queries branch off from that conversation. Their
answers and labels stay out of the observed history.

Use Python 3.12 on Ada. These commands create a private environment with the
vLLM version used here. Generate or transfer the OCR, diagram and noise PNGs and
their manifests, then run the benchmark from the repository root.

~~~bash
ENV="$HOME/bvlm14-ada-env"
python3 -m venv --without-pip "$ENV"
python3 -m pip --python "$ENV/bin/python" install pip 'vllm==0.34.0'
export BVLM_PYTHON="$ENV/bin/python"
export PATH="$ENV/bin:$PATH"
~~~

If a compute node reports that Python.h is missing, use the matching Python 3.12
headers from the Ada login node. Export the include paths before calling sbatch
so Slurm passes them through to the compiler.

~~~bash
HEADERS="$HOME/bvlm14-python-headers"
mkdir -p "$HEADERS/x86_64-linux-gnu"
cp -a /usr/include/python3.12 "$HEADERS/"
cp -a /usr/include/x86_64-linux-gnu/python3.12 "$HEADERS/x86_64-linux-gnu/"
export C_INCLUDE_PATH="$HEADERS/python3.12:$HEADERS"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
~~~

Keep the model revision fixed across jobs. For a small pipeline check, the
optional smoke test uses two users, two offers and three observation counts.

~~~bash
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
FULL=data/generated/bvlm-14-ada-qwen35-9b-bf16
mkdir -p data/generated/ada-slurm

# Optional smoke test
sbatch --parsable --time=00:30:00 benchmarks/visual/run_flight_ada.sh \
  --revision "$REV" --out data/generated/bvlm-14-ada-smoke
~~~

The full benchmark uses 32 shards with up to eight simultaneous GPU jobs.
Adjust concurrency to the available resources. Preserve the shard count and
settings when resubmitting a failed shard.

~~~bash
sbatch --parsable --time=04:00:00 --array=0-31%8 benchmarks/visual/run_flight_ada.sh \
  --revision "$REV" --full --shards 32 --out "$FULL"
~~~

The full grid holds 624 users × 100 offers × 6 observation counts for each
format. The completed Ada run includes 1,497,600 validated predictions across
all four formats. Open the [metrics table](benchmarks/visual/results/ada-qwen35-9b-bf16/metrics.csv)
or the [shareable plot and round-by-round table](benchmarks/visual/results/ada-qwen35-9b-bf16/comparison.png).

Completed round files are checkpoints. Run the summarizer after all
jobs finish. It verifies exact case coverage and creates one merged JSONL file
per format for the team evaluator. The report contains the measured results
for the four input formats from that run.

~~~bash
python3 benchmarks/visual/summarize_flight_ada.py --run "$FULL" \
  --out benchmarks/visual/results/ada-qwen35-9b-bf16 --plot
~~~

Install matplotlib in the analysis environment for `--plot`. The report
directory is part of the repository and holds the compact `metrics.csv`,
`summary.json` and `accuracy.png`. The full report also generates
`comparison.png` with a table of scores by round. The ignored run directory
holds the logs, round checkpoints and merged predictions. Metrics use the
stored held-out choice as the target. Unparseable responses stay in the
denominator and are exported as `rec: null`. The merged files use the
[BVLM-9 evaluator](https://github.com/c-tutsch/Bayesian-VLMs/tree/eren-bvlm-9-bayesian-reference/bayesian_reference)
prediction schema.

The optional smoke test and local runner
(`python3 benchmarks/visual/run_flight_pair.py --help`) can be used to check
the pipeline. Keep their outputs under `data/generated/`. Figures and metrics
from `--dry-run` are synthetic.
