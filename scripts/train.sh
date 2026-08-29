#!/usr/bin/env bash
# Launch one SDFT run with a record in experiments/runs/<YYYY-MM-DD_label>/.
#
#   PROFILE=kakao|a6000  scripts/train.sh <tooluse|science|kkp> <label> [extra main.py args...]
#
#   LR=5e-5 EPOCHS=1 SAVE_STEPS=30 EVAL_STEPS=<profile: kakao 10, a6000 30> MODEL=Qwen/Qwen3-4B WANDB_PROJECT=sdft   (env overrides)
#   e.g.  PROFILE=a6000 LR=1e-4 scripts/train.sh kkp lr1e-4 --group kkp-lr-sweep --idea opsd-kkp
#
# PROFILE sets the colocated vLLM memory share, the in-training eval sample count and (a6000)
# gradient checkpointing; the dataset sets prompt/completion lengths (kkp teacher prompts reach
# 4558 tokens: 8192/8192 on kakao, 5120/4096 on a6000).
set -euo pipefail
# Launcher of record for the run (main.py copies it into the record). An outer script in
# experiments/launch/ sets these first and wins; direct calls fall back to this file.
export SDFT_LAUNCH_CMD="${SDFT_LAUNCH_CMD:-$(printf '%q ' "$0" "$@")}"
export SDFT_LAUNCHER="${SDFT_LAUNCHER:-$(cd "$(dirname "$0")" && pwd)/$(basename "$0")}"
cd "$(dirname "$0")/.."

DATASET="${1:?dataset (tooluse|science|kkp)}"; LABEL="${2:?label}"; shift 2
PROFILE="${PROFILE:-kakao}"

case "$PROFILE" in
  kakao) VLLM_MEM=0.3; EVAL_N=300; EVAL_EVERY=10; GC=() ;;                        # 80 GB class
  a6000) VLLM_MEM=0.4; EVAL_N=100; EVAL_EVERY=30; GC=(--gradient_checkpointing) ;;  # 48 GB: engine sleeps during the loss phase, so 0.4 costs training nothing; generation was KV-starved at 0.25
  *) echo "unknown PROFILE=$PROFILE (kakao|a6000)"; exit 2 ;;
esac

case "$DATASET" in
  kkp)     if [ "$PROFILE" = a6000 ]; then PLEN=5120; CLEN=4096; EVAL_TOK=4096   # max teacher prompt 4558 tokens; CoTs ~1.1k
           else PLEN=8192; CLEN=8192; EVAL_TOK=8192; fi ;;
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
  --vllm_gpu_memory_utilization "$VLLM_MEM" "${GC[@]}" \
  --eval_steps "${EVAL_STEPS:-$EVAL_EVERY}" --eval_num_samples "$EVAL_N" --eval_max_new_tokens "$EVAL_TOK" \
  --wandb_project "${WANDB_PROJECT:-sdft}" \
  "$@"
