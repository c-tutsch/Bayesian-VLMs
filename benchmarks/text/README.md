# Bayesian reasoning benchmark

Reproduces the flight recommendation benchmark (Fig. 2) from Qiu et al. 2026, _Bayesian teaching enables probabilistic reasoning in large language models_ (Nat. Commun.), as a reference for testing newer models.

## Setup (cluster)

```bash
conda create -n bayes-bench python=3.11 && conda activate bayes-bench
pip install -r requirements.txt   # vLLM, Linux + CUDA only
```

## What it does

For each of the 624 simulated users in `data/eval/interaction/flight.jsonl`, the model plays 5 rounds of "Which flight is the best option?". After each round it gets feedback based on its own answer (prompt wording follows Supp. Table H3). After 0..5 rounds (`n_obs`), the conversation branches off once for each of the 100 held-out option sets in `data/eval/heldout/flight.json` (Supp. Fig. A1). These branches never go back into the main conversation. Decoding is greedy, and no system prompt is set, so each model uses its chat template's default.

Output for `--out X.jsonl`:

- `X.jsonl`: `{"user", "n_obs", "set", "rec"}` per line, 624 × 6 × 100 = 374,400 lines, the input for the team's `evaluate_predictions`
- `X.raw.jsonl`: every model answer, for checking how answers were parsed
- `X.stats.json`: parse failure rate and accuracy per `n_obs`

Without `--thinking`, the assistant turn is pre-filled with `The best option is Flight` (the answer format from Supp. Table H3), and the model only completes the number. With `--thinking`, the model generates freely and a regex parser extracts the answer. Unparseable answers get a seeded random flight (counted in the stats). The model's visible answer, with thinking stripped, stays in the history. vLLM prefix caching reuses each user's history across its 100 held-out queries.

## Run

```bash
# 1. Quick test on an interactive GPU node (3 users = 1,815 prompts, a few minutes)
bash run_interactive.sh
conda activate bayes-bench
python run_flight.py --limit-users 3 --out results/test-run.jsonl
# check results/test-run.stats.json (parse_fail should be ~0) and skim results/test-run.raw.jsonl

# 2. Full run as a batch job
mkdir -p slurm_logs && sbatch run_flight.sh --model Qwen/Qwen2.5-7B-Instruct --out results/flight_qwen2.5-7b.jsonl
```

| Option             | Meaning                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| `--model`          | Hugging Face id (default `Qwen/Qwen2.5-7B-Instruct`)                           |
| `--limit-users`    | First N users only, for testing (the output then fails `evaluate_predictions`) |
| `--thinking`       | Reasoning on (default off)                                                     |
| `--max-new-tokens` | Output limit per answer (default 256, raise for thinking models)               |
| `--max-model-len`  | vLLM context length (default 8192)                                             |
| `--tp`             | Tensor-parallel GPUs, e.g. `--tp 2` with `sbatch --gres=gpu:2`                 |
| `--dry-run`        | Fake model, no GPU: checks only the pipeline and output format                 |

One L40S (48 GB) fits models up to about 20B parameters in bf16. Qwen2.5-32B needs `--tp 2`.

Paper reference (seed 0 data): Qwen2.5-7B 37 → 37 % (`n_obs` 1 → 5), Bayesian Assistant ≈ 57 → 80 %.
