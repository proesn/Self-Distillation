#!/usr/bin/env bash
# 2026-08-30 — eval-kkp-baseline
#
# What this launch tests:
#   The precise kkp numbers behind the flat in-training curve of 2026-08-29_sdft-kkp (57 → 59, n=100):
#   the final checkpoint (step 300) and the untouched base model, both on all 300 eval puzzles, greedy,
#   4096 new tokens, seed 42. Two eval records: eval-kkp-2026-08-29_sdft-kkp-s300 and eval-kkp-base.
#   Run this BEFORE launching training — both need the whole GPU. ~20 min each.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"
run() { echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"; scripts/eval.sh "$@" 2>&1 | tee -a "$LOG"; echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"; }

run kkp 2026-08-29_sdft-kkp 300 --max_new_tokens 4096 "$@"
run kkp base                    --max_new_tokens 4096 "$@"
