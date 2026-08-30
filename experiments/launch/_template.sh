#!/usr/bin/env bash
# __DATE__ — __LABEL__
#
# What this launch tests (one or two lines, your words):
#   ...
#
# Every launch is one script here, tracked in git. main.py copies this file into the run's
# record (experiments/runs/<id>/launcher.sh) and stores the command you typed, so the run
# and the script that produced it can never drift apart. Re-run a run: `python -m explog rerun <id>`.
# Start it detached:  nohup <this file> > /dev/null 2>&1 &     (it logs itself to logs/)
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"

# run <dataset> <label> [args]: one training run; the chain continues even if this one fails
# (a run is judged by its record status, not by the exit code of the vLLM/NCCL teardown).
run() {
  echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"
  scripts/train.sh "$@" 2>&1 | tee -a "$LOG"
  echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"
}

export PROFILE=__PROFILE__          # a6000 (48 GB) | kakao (80 GB)
export LR=5e-5 EPOCHS=1             # SAVE_STEPS=30 EVAL_STEPS=<profile default> MODEL=Qwen/Qwen3-4B WANDB_PROJECT=sdft

run __DATASET__ __LABEL__ --group __LABEL__ "$@"

python -m explog table
