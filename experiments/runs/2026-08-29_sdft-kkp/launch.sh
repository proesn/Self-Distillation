#!/usr/bin/env bash
# Re-run of 2026-08-29_sdft-kkp — generated at launch by sdft/runlog.py.
#   typed:   experiments/launch/2026-08-29_sdft-3ds-a6000.sh 
#   host:    jong    cwd: /home/user/jjkim/Self-Distillation
#   python:  /home/user/.envs/jjkim_distillation/bin/python
# Usage: bash experiments/runs/2026-08-29_sdft-kkp/launch.sh [--name <label>] [extra main.py args]
#   Runs at the recorded code state. If your checkout differs (other commit, dirty tree, or the run
#   had a code.patch), it runs inside a throwaway worktree under .rerun-worktrees/ — your checkout
#   is never touched. SDFT_RERUN_SAME_CODE=0 runs your current code instead.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHA=cf155082d0a64254885366c36adc8e71f06902cd
if [ "${SDFT_RERUN_SAME_CODE:-1}" = 1 ] && { [ "$(git rev-parse HEAD)" != "$SHA" ] || [ -n "$(git status --porcelain)" ] || [ -s "$HERE/code.patch" ]; }; then
  WT="$ROOT/.rerun-worktrees/${SHA:0:10}"
  [ -d "$WT" ] || git worktree add --detach --quiet "$WT" "$SHA"
  git -C "$WT" checkout --quiet -- .
  [ -s "$HERE/code.patch" ] && git -C "$WT" apply "$HERE/code.patch"
  echo "[launch.sh] running in worktree $WT at $SHA$([ -s "$HERE/code.patch" ] && echo ' + code.patch')"
  cd "$WT"
fi
export CONDA_DEFAULT_ENV=/home/user/.envs/jjkim_distillation
export CUDA_VISIBLE_DEVICES=0
export EPOCHS=1
export HF_HOME=/home/user/jjkim/.cache/huggingface
export LR=5e-5
export PROFILE=a6000
export SDFT_LAUNCHER=/home/user/jjkim/Self-Distillation/experiments/launch/2026-08-29_sdft-3ds-a6000.sh
export SDFT_LAUNCH_CMD='experiments/launch/2026-08-29_sdft-3ds-a6000.sh '
export TORCH_HOME=/home/user/jjkim/.cache/torch
export WANDB_PROJECT=self-distillation
export SDFT_RUNS_DIR="${SDFT_RUNS_DIR:-$ROOT/experiments/runs}"
export SDFT_CHECKPOINTS_DIR="${SDFT_CHECKPOINTS_DIR:-$ROOT/checkpoints}"
exec "${PYTHON:-/home/user/.envs/jjkim_distillation/bin/python}" main.py --dataset_name kkp --name sdft-kkp-rerun --model_name Qwen/Qwen3-4B --use_lora --save_lora_adapter_only --learning_rate 5e-5 --num_train_epochs 1 --save_steps 30 --max_prompt_length 5120 --max_completion_length 4096 --vllm_gpu_memory_utilization 0.4 --gradient_checkpointing --eval_steps 30 --eval_num_samples 100 --eval_max_new_tokens 4096 --wandb_project self-distillation --group sdft-3ds --tags baseline "$@"
