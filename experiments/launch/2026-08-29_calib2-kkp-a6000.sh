#!/usr/bin/env bash
# 2026-08-29 — calib2-kkp-a6000
#
# What this launch tests:
#   Throughput re-check after the a6000 profile change (vLLM share 0.25 → 0.4). calib ran at 277 s/step;
#   6 optimizer steps, evals at 0 and 5 on 20 samples, checkpoint at 5. Judged `invalid` (calibration).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"
run() { echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"; scripts/train.sh "$@" 2>&1 | tee -a "$LOG"; echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"; }

export PROFILE=a6000
export EPOCHS=0.02 SAVE_STEPS=5 EVAL_STEPS=5

run kkp calib2 --eval_num_samples 20 --tags calib "$@"
