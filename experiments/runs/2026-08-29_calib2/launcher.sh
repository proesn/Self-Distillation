#!/usr/bin/env bash
# 2026-08-29 — calib2-kkp-a6000
#
# What this launch tests:
#   Throughput re-check after the a6000 profile change (vLLM share 0.25 → 0.4). calib (2026-08-29_calib)
#   ran at 277 s/step and ~5 min per 20-sample eval — suspected KV-cache starvation. 6 optimizer steps,
#   evals at 0 and 5 on 20 samples, checkpoint at 5; events now carry gpu_mem_peak_gb. Judge `invalid`.
#   Compare: python -m explog compare calib calib2
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export SDFT_LAUNCHER="$(realpath "$0")" SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
mkdir -p logs

export PROFILE=a6000
export EPOCHS=0.02 SAVE_STEPS=5 EVAL_STEPS=5

scripts/train.sh kkp calib2 --eval_num_samples 20 --tags calib "$@" 2>&1 | tee "logs/2026-08-29_$(basename "$0" .sh).log"
