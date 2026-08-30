#!/usr/bin/env bash
# 2026-08-29 — sdft-3ds-a6000
#
# What this launch tests:
#   SDFT baseline on the three datasets with Qwen3-4B + LoRA r64, EMA teacher 0.01, forward KL,
#   lr 5e-5 — kkp 1 epoch (310 steps), science 2 epochs (168), tooluse 2 epochs (252), sequential
#   on one A6000. Periodic eval every 30 steps on 100 samples.
#
# History: run once on 2026-08-29 19:01 (record 2026-08-29_sdft-kkp, finished 08-30 11:13, 16 h).
#   The chain stopped after kkp because the original script used `set -e` and the vLLM/NCCL
#   teardown returned non-zero. science + tooluse were launched separately:
#   experiments/launch/2026-08-30_sdft-2ds-a6000.sh. Do NOT re-run this file — it would repeat kkp.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"
run() { echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"; scripts/train.sh "$@" 2>&1 | tee -a "$LOG"; echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"; }

export PROFILE=a6000 LR=5e-5

EPOCHS=1 run kkp     sdft-kkp     --group sdft-3ds --tags baseline "$@"
EPOCHS=2 run science sdft-science --group sdft-3ds --tags baseline "$@"
EPOCHS=2 run tooluse sdft-tooluse --group sdft-3ds --tags baseline "$@"

python -m explog table
