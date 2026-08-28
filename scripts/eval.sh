#!/usr/bin/env bash
# Standalone evaluation of a run's checkpoint; the result is appended to that run's results.json.
#
#   scripts/eval.sh <tooluse|science|kkp> <run_id> [checkpoint-step] [extra eval args...]
#
#   Defaults: latest checkpoint under checkpoints/<run_id>/, base model from the run record,
#   greedy decoding, seed 42. Cross-dataset (forgetting) evals: pass a dataset other than the
#   one the run trained on — it lands in the same results.json and in the INDEX's standalone table.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?dataset}"; RUN_ID="${2:?run_id}"; shift 2
RUN_DIR="experiments/runs/$RUN_ID"
[ -f "$RUN_DIR/run.json" ] || { echo "no run record at $RUN_DIR"; exit 2; }

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then STEP="$1"; shift; CKPT="checkpoints/$RUN_ID/checkpoint-$STEP"
else CKPT="$(ls -d checkpoints/"$RUN_ID"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"; fi
[ -d "$CKPT" ] || { echo "no checkpoint found for $RUN_ID"; exit 2; }

MODEL="$(python -c "import json;print(json.load(open('$RUN_DIR/run.json'))['model']['name'])")"

case "$DATASET" in
  kkp)     EXTRA=(--max_new_tokens 8192 --max_model_len 16384) ;;
  science) EXTRA=(--max_new_tokens 2048 --max_model_len 4096) ;;
  tooluse) EXTRA=(--max_new_tokens 1024) ;;
  *) echo "unknown dataset $DATASET"; exit 2 ;;
esac

python "eval_$DATASET.py" --model_path "$MODEL" --adapter_path "$CKPT" --run_dir "$RUN_DIR" \
  --output_dir "$CKPT/eval_$DATASET" --seed 42 --temperature 0.0 "${EXTRA[@]}" "$@"
python -m explog table
