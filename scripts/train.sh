#!/usr/bin/env bash
# Launch one SDFT run with a record in experiments/runs/<YYYY-MM-DD_label>/.
#
#   PROFILE=kakao|a6000  scripts/train.sh <tooluse|science|kkp> <label> [extra main.py args...]
#
#   LR=5e-5 EPOCHS=1 SAVE_STEPS=30 EVAL_STEPS=10 MODEL=Qwen/Qwen3-4B WANDB_PROJECT=sdft   (env overrides)
#   e.g.  PROFILE=a6000 LR=1e-4 scripts/train.sh kkp lr1e-4 --group kkp-lr-sweep --idea opsd-kkp
#
# PROFILE sets the colocated vLLM memory share and the in-training eval sample count;
# the dataset sets prompt/completion lengths (kkp teacher prompts reach 4558 tokens, so 8192).
set -euo pipefail
# Record what was typed so the run's launch.sh can reproduce it (main.py reads these).
export SDFT_LAUNCH_CMD="$(printf '%q ' "$0" "$@")"
export SDFT_LAUNCHER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."

DATASET="${1:?dataset (tooluse|science|kkp)}"; LABEL="${2:?label}"; shift 2
PROFILE="${PROFILE:-kakao}"

case "$PROFILE" in
  kakao) VLLM_MEM=0.3;  EVAL_N=300 ;;   # 80 GB class
  a6000) VLLM_MEM=0.2;  EVAL_N=100 ;;   # 48 GB — tighter engine share, smaller periodic eval
  *) echo "unknown PROFILE=$PROFILE (kakao|a6000)"; exit 2 ;;
esac

case "$DATASET" in
  kkp)     PLEN=8192; CLEN=8192; EVAL_TOK=8192 ;;
  science) PLEN=2048; CLEN=2048; EVAL_TOK=2048 ;;
  tooluse) PLEN=2048; CLEN=1024; EVAL_TOK=1024 ;;
  *) echo "unknown dataset $DATASET"; exit 2 ;;
esac

python main.py \
  --dataset_name "$DATASET" --name "$LABEL" \
  --model_name "${MODEL:-Qwen/Qwen3-4B}" \
  --use_lora --save_lora_adapter_only \
  --learning_rate "${LR:-5e-5}" --num_train_epochs "${EPOCHS:-1}" --save_steps "${SAVE_STEPS:-30}" \
  --max_prompt_length "$PLEN" --max_completion_length "$CLEN" \
  --vllm_gpu_memory_utilization "$VLLM_MEM" \
  --eval_steps "${EVAL_STEPS:-10}" --eval_num_samples "$EVAL_N" --eval_max_new_tokens "$EVAL_TOK" \
  --wandb_project "${WANDB_PROJECT:-sdft}" \
  "$@"
