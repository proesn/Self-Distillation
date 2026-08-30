#!/usr/bin/env bash
# 2026-08-30 — sdft-tooluse-a6000
#
# What this launch tests:
#   The tooluse half of the SDFT baseline (group sdft-3ds), same settings as 2026-08-29_sdft-kkp and
#   2026-08-30_sdft-science: Qwen3-4B + LoRA r64, EMA teacher 0.01, forward KL, lr 5e-5, 2 epochs
#   (252 steps), periodic eval every 30 steps on 100 samples. This is the tooluse leg of
#   2026-08-30_sdft-2ds-a6000.sh, which never started there (GPU pre-flight refused 5 s after the
#   science process exited, while the driver was still releasing its memory).
#
#   Start:     bash experiments/launch/2026-08-30_sdft-tooluse-a6000.sh   (in a tmux pane)
#   Progress:  tail -f logs/2026-08-30_sdft-tooluse-a6000.log   ·   python -m explog table
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"
run() { echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"; scripts/train.sh "$@" 2>&1 | tee -a "$LOG"; echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"; }

export PROFILE=a6000 LR=5e-5

EPOCHS=2 run tooluse sdft-tooluse --group sdft-3ds --tags baseline "$@"

python -m explog table
