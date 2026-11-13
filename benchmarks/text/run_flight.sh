#!/bin/bash
#SBATCH --job-name=flight
#SBATCH --partition=L40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j-%x.out
#SBATCH --error=slurm_logs/%j-%x.err
# Usage: mkdir -p slurm_logs && sbatch run_flight.sh --model Qwen/Qwen2.5-7B-Instruct --out results/flight_qwen2.5-7b.jsonl
# 2 GPUs (e.g. Qwen2.5-32B): sbatch --gres=gpu:2 run_flight.sh --tp 2 --model Qwen/Qwen2.5-32B-Instruct --out ...
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bayes-bench
cd "$SLURM_SUBMIT_DIR"
python run_flight.py "$@"
