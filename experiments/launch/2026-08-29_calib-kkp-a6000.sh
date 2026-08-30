#!/usr/bin/env bash
# 2026-08-29 — calib-kkp-a6000
#
# What this launch tests:
#   Pipeline calibration on the A6000: vLLM colocate + LoRA + 5120/4096 windows + gradient
#   checkpointing + periodic eval + record writing. ~15 optimizer steps, checkpoint at 10,
#   20-sample eval every 5 steps. Judged `invalid` (calibration, not an experiment).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"
run() { echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"; scripts/train.sh "$@" 2>&1 | tee -a "$LOG"; echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"; }

export PROFILE=a6000
export EPOCHS=0.05 SAVE_STEPS=10 EVAL_STEPS=5

run kkp calib --eval_num_samples 20 --tags calib "$@"
