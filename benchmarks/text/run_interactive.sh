#!/bin/bash
# Development only: 1 L40S GPU shell for 4h. `exit` when done.
# (Does not run for 4h, just creates a session for 4h.)
srun -p L40s --gres=gpu:1 --cpus-per-task=16 --mem=32GB \
  --time=04:00:00 --job-name=interactive --pty bash -i
