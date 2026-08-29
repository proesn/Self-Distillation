#!/usr/bin/env bash
# __DATE__ — __LABEL__
#
# What this launch tests (one or two lines, your words):
#   ...
#
# Every launch is one script here, tracked in git. main.py copies this file into the run's
# record (experiments/runs/<id>/launcher.sh) and stores the command you typed, so the run
# and the script that produced it can never drift apart. Re-run a run: `python -m explog rerun <id>`.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
mkdir -p logs

export PROFILE=__PROFILE__          # a6000 (48 GB) | kakao (80 GB)
export LR=5e-5 EPOCHS=1             # SAVE_STEPS=30 EVAL_STEPS=10 MODEL=Qwen/Qwen3-4B WANDB_PROJECT=sdft

scripts/train.sh __DATASET__ __LABEL__ --group __LABEL__ "$@" 2>&1 | tee "logs/__DATE___$(basename "$0" .sh).log"
