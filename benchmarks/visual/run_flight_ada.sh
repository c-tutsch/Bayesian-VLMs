#!/bin/bash
#SBATCH --job-name=bvlm14
#SBATCH --partition=L40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=data/generated/ada-slurm/%x-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${BVLM_PYTHON:?Set BVLM_PYTHON to the Python executable with vLLM installed}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    set -- "$@" --shard "$SLURM_ARRAY_TASK_ID"
fi
exec "$BVLM_PYTHON" -u benchmarks/visual/run_flight_ada.py "$@"
