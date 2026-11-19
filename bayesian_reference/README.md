# Bayesian Assistant — Reference Module

## What this is

A NumPy reimplementation of the Bayesian Assistant and its evaluation metrics for the flight recommendation task in Qiu et al., "Bayesian teaching enables probabilistic reasoning in large language models" (Nature Communications 2026).
It is the ground-truth reference for LLM/VLM evaluations: a model run writes a predictions file, and this module scores it.
It contains no model code.

## Setup
Run all commands from this folder (bayesian_reference/).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Download the data from Zenodo (<https://zenodo.org/records/17677329>) and unzip it to `./data`. The `data/` folder is gitignored. Tests that need the data are skipped if it is missing. (but scripts need the data)

## Quick start

```bash
pytest -q
python scripts/run_reference.py
python scripts/evaluate_file.py PREDICTIONS.jsonl --users 0-49 --n-obs 1,5 --sets 0-99 --name MODEL
```

`run_reference.py` writes the Bayesian and oracle reference curves to `outputs/`. `evaluate_file.py` prints success, agreement with the Bayesian assistant and the fail rate for each `n_obs`, next to the reference values on the same subset.

## Predictions format

JSONL with one prediction per line and exactly these keys:

```json
{"user": 17, "n_obs": 3, "set": 42, "rec": 0}
```

- `user`: 0-based row in `data/eval/interaction/flight.jsonl` (0..623).
- `n_obs`: number of observed interaction rounds (0..5; 0 means no observations).
- `set`: index into `all_options` of `data/eval/heldout/flight.json` (0..99).
- `rec`: 0-based recommended flight (Flight 1 -> 0, Flight 2 -> 1, Flight 3 -> 2). Use `"rec": null` when the model's answer could not be parsed; it counts as a failure.

A full file has 624 × 6 × 100 = 374,400 lines. For a subset, declare it with `--users`, `--n-obs` and `--sets` (e.g. `0-49`, `1,5`, `all`). Every declared combination must appear exactly once, and lines outside the declared subset are errors.

## Common mistakes

- **1-based `rec`.** "Flight 1" must be written as `0`, not `1`.
- **User order.** `user` is the row number in the interaction file, not a position in any shuffled or re-sorted list.
- **Set order.** `set` follows the order of `all_options` in the held-out file; do not re-sort the sets or the options inside a set.

## Comparing runs

Every command takes the same `--users`, `--n-obs` and `--sets` flags as `evaluate_file.py` (default `all`, i.e. full files), and exits with code 1 on any validation error. Run files are passed as `NAME=PATH`. The `+-` values are the standard error of the mean over users, which is not comparable to the paper's error bars.

1. Score a single file:

   ```bash
   python scripts/evaluate_file.py path/to/run.jsonl --name MODEL
   ```

2. Compare several runs with each other and with the Bayesian and oracle references. The command prints success, agreement and fail rate for each run at each `n_obs`. It can also write the table as Markdown, CSV or JSON, a line plot and a bar plot at rounds 1 and 5. `--note` sets the caption under the plots, for example to say that this is a single seed, non-thinking, with prefilled answers:

   ```bash
   python scripts/compare_runs.py model-a=path/to/run_a.jsonl model-b=path/to/run_b.jsonl \
       --table-md outputs/compare.md --table-csv outputs/compare.csv --table-json outputs/compare.json \
       --plot outputs/compare.png --plot-rounds outputs/compare_rounds.png --note "single seed, non-thinking"
   ```

3. Paired comparison of two files on the same `(user, n_obs, set)` cases, for example text vs OCR input. The command prints success of A and B, the difference B - A with a paired SE over users, agreement per file, the share of changed recommendations and the direction of each change (correct->wrong, wrong->correct, wrong->wrong, null involved). If the two files do not cover the same cases it stops with an error. With `--intersect` it uses only the shared cases and reports how many there are:

   ```bash
   python scripts/paired_compare.py text=path/to/text_run.jsonl ocr=path/to/ocr_run.jsonl --table-md outputs/paired.md
   python scripts/paired_compare.py text=path/to/text_run.jsonl ocr=path/to/ocr_run.jsonl --users 0-49 --intersect
   ```

4. Deviation from approximate paper values. For each run with a matching name, and for `bayesian` / `oracle`, the command reports `deviation = ours - paper` and `deviation / SE`. This is a report, not a pass/fail test. The JSON format is documented in `scripts/compare_runs.py`. `docs/paper_approx_example.json` holds approximate values read from the paper's figures for `qwen2.5-7b`, `qwen2.5-32b` and `bayesian`:

   ```bash
   python scripts/compare_runs.py qwen2.5-7b=path/to/run_7b.jsonl qwen2.5-32b=path/to/run_32b.jsonl \
       --paper-approx docs/paper_approx_example.json
   ```

## Validation

Expected success over 624 users × 100 held-out sets:

| n_obs | Bayesian assistant (ours) | Paper (approx.) |
|------:|--------------------------:|----------------:|
| 0     | 0.333                     | 0.33            |
| 1     | 0.569                     | 0.58            |
| 5     | 0.808                     | 0.81            |

The oracle (scores with the user's true preferences) reaches 0.973, not 1.0, because some held-out sets contain tied best options.

## Known limitations

- The tie-breaking rule in the likelihood and the hypothesis that the paper's training data was generated by posterior sampling are inferences from the data. The paper does not confirm them.
- Results use a single seed of the released eval data; the paper averages over three seeds.

## Repo layout

```
bayes_ref/   core library: world, user choice, Bayesian and oracle assistants, evaluation, plotting
scripts/     command-line entry points and read-only data diagnostics
tests/       pytest suite (data-dependent tests skip without ./data)
data/        Zenodo data (gitignored, not in the repo)
outputs/     generated curves, plots and example prediction files (gitignored)
```
