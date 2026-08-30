#!/usr/bin/env bash
# 2026-08-29 — sdft-3ds-a6000
#
# What this launch tests:
#   SDFT baseline on the three datasets with Qwen3-4B + LoRA r64, EMA teacher 0.01, forward KL,
#   lr 5e-5 — kkp 1 epoch (310 steps), science 2 epochs (168), tooluse 2 epochs (252), sequential
#   on one A6000. Periodic eval every 10 steps on 100 samples. The reference point every later
#   variant (frozen teacher, OPSD clamp, reverse KL) is compared against.
#
#   Detached:  nohup experiments/launch/2026-08-29_sdft-3ds-a6000.sh > /dev/null 2>&1 &
#   Progress:  tail -f logs/2026-08-29_sdft-3ds-a6000.log   ·   python -m explog table
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
mkdir -p logs
LOG="logs/2026-08-29_$(basename "$0" .sh).log"

export PROFILE=a6000 LR=5e-5

EPOCHS=1 scripts/train.sh kkp     sdft-kkp     --group sdft-3ds --tags baseline "$@" 2>&1 | tee -a "$LOG"
EPOCHS=2 scripts/train.sh science sdft-science --group sdft-3ds --tags baseline "$@" 2>&1 | tee -a "$LOG"
EPOCHS=2 scripts/train.sh tooluse sdft-tooluse --group sdft-3ds --tags baseline "$@" 2>&1 | tee -a "$LOG"

python -m explog table
