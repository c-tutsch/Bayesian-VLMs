#!/bin/bash
# Development only: 1 L40S GPU shell for 30min. `exit` when done.
# (Does not run for 30min, just creates a session for 30min.)
srun -p L40s --gres=gpu:1 --cpus-per-task=8 --mem=16GB \
  --time=00:30:00 --job-name=interactive --pty bash -i
