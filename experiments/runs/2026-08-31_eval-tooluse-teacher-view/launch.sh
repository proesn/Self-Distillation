#!/usr/bin/env bash
# Re-run of 2026-08-31_eval-tooluse-teacher-view — generated at launch by sdft/runlog.py.
#   typed:   experiments/launch/2026-08-31_eval-teacher-view-base.sh 
#   host:    jong    cwd: /home/user/jjkim/Self-Distillation
#   python:  /home/user/jjkim/.envs/jjkim_distillation/bin/python
# Usage: bash experiments/runs/2026-08-31_eval-tooluse-teacher-view/launch.sh [--name <label>] [extra main.py args]
#   Runs at the recorded code state. If your checkout differs (other commit, dirty tree, or the run
#   had a code.patch), it runs inside a throwaway worktree under .rerun-worktrees/ — your checkout
#   is never touched. SDFT_RERUN_SAME_CODE=0 runs your current code instead.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHA=3a46b6ad77f7e2a42f3b421747232912de0eea40
if [ "${SDFT_RERUN_SAME_CODE:-1}" = 1 ] && { [ "$(git rev-parse HEAD)" != "$SHA" ] || [ -n "$(git status --porcelain)" ] || [ -s "$HERE/code.patch" ]; }; then
  WT="$ROOT/.rerun-worktrees/${SHA:0:10}"
  [ -d "$WT" ] || git worktree add --detach --quiet "$WT" "$SHA"
  git -C "$WT" checkout --quiet -- .
  [ -s "$HERE/code.patch" ] && git -C "$WT" apply "$HERE/code.patch"
  echo "[launch.sh] running in worktree $WT at $SHA$([ -s "$HERE/code.patch" ] && echo ' + code.patch')"
  cd "$WT"
fi
export CONDA_DEFAULT_ENV=/home/user/jjkim/.envs/jjkim_distillation
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/user/jjkim/.cache/huggingface
export SDFT_LAUNCHER=/home/user/jjkim/Self-Distillation/experiments/launch/2026-08-31_eval-teacher-view-base.sh
export SDFT_LAUNCH_CMD='experiments/launch/2026-08-31_eval-teacher-view-base.sh '
export TORCH_HOME=/home/user/jjkim/.cache/torch
export WANDB_PROJECT=self-distillation
export SDFT_RUNS_DIR="${SDFT_RUNS_DIR:-$ROOT/experiments/runs}"
export SDFT_CHECKPOINTS_DIR="${SDFT_CHECKPOINTS_DIR:-$ROOT/checkpoints}"
exec "${PYTHON:-/home/user/jjkim/.envs/jjkim_distillation/bin/python}" eval_tooluse.py --model_path Qwen/Qwen3-4B --name eval-tooluse-teacher-view-rerun --output_dir eval_results/eval-tooluse-teacher-view --teacher_view --num_samples 300 --seed 42 --temperature 0.0 --max_new_tokens 1024 "$@"
