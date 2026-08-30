#!/usr/bin/env bash
# 2026-08-31 — eval-teacher-view-base
#
# What this launch tests:
#   What the SDFT teacher signal contains, per dataset, before any training: the base model
#   (Qwen3-4B, no adapter) scored under the trainer's teacher prompt (question + that item's
#   demonstration + "answer with a response of your own") versus under the plain prompt, on the same
#   300 seeded training items. Three eval records: eval-{science,tooluse,kkp}-teacher-view, each with
#   teacher_accuracy / student_accuracy / gap and the paired counts. Greedy, seed 42; kkp at 4096 new
#   tokens (a6000 window). Reference: the SDFT paper's ToolAlpaca check (base 42% → teacher 100%).
#
#   Start:     bash experiments/launch/2026-08-31_eval-teacher-view-base.sh   (in a tmux pane)
#   Progress:  tail -f logs/2026-08-31_eval-teacher-view-base.log   ·   python -m explog table
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/$(basename "$0" .sh).log"
run() { echo "[launch] $(date '+%F %T') start: $*" | tee -a "$LOG"; scripts/eval.sh "$@" 2>&1 | tee -a "$LOG"; echo "[launch] $(date '+%F %T') exit ${PIPESTATUS[0]}: $*" | tee -a "$LOG"; }

run science teacher 300 "$@"
run tooluse teacher 300 "$@"
run kkp     teacher 300 --max_new_tokens 4096 "$@"
