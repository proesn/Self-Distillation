#!/usr/bin/env bash
# 2026-08-29 — calib-kkp-a6000
#
# What this launch tests:
#   Pipeline calibration on the A6000: vLLM colocate + LoRA + 5120/4096 windows + gradient
#   checkpointing + periodic eval + record writing. ~15 optimizer steps, checkpoint at 10,
#   20-sample eval every 5 steps. Judge as `invalid` afterwards (it is not an experiment).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
mkdir -p logs

export PROFILE=a6000
export EPOCHS=0.05 SAVE_STEPS=10 EVAL_STEPS=5

scripts/train.sh kkp calib --eval_num_samples 20 --tags calib "$@" 2>&1 | tee "logs/2026-08-29_$(basename "$0" .sh).log"

# then, by hand:
#   python -m explog table && python -m explog show calib
#   scripts/eval.sh kkp <run id> 10 --num_samples 20
#   python -m explog note <run id> --validity invalid --reason "calibration, 15 steps" --verdict "pipeline check"
