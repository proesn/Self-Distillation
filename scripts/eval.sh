#!/usr/bin/env bash
# Standalone evaluation as a record of its own (experiments/runs/<date>_<label>/, kind: eval),
# linked to the training run it targets.
#
#   scripts/eval.sh <tooluse|science|kkp> <run_id> [checkpoint-step] [extra eval args...]   # a training run's checkpoint
#   scripts/eval.sh <tooluse|science|kkp> base [extra eval args...]                          # the base model (MODEL env, default Qwen/Qwen3-4B)
#   scripts/eval.sh <tooluse|science|kkp> teacher [n] [extra eval args...]                   # teacher view: base + teacher prompt vs base + plain prompt on n (300) training items
#
#   Defaults: latest checkpoint under checkpoints/<run_id>/, base model from the run record,
#   greedy decoding, seed 42, full eval set. Responses are written next to the checkpoint
#   (untracked); metrics + per-sample scores go into the record (tracked).
#   Label: LABEL=<name> env, default eval-<dataset>-<run_id or base>[-s<step>].
set -uo pipefail
cd "$(dirname "$0")/.."
export SDFT_LAUNCH_CMD="${SDFT_LAUNCH_CMD:-$(printf '%q ' "$0" "$@")}"
export SDFT_LAUNCHER="${SDFT_LAUNCHER:-$(cd "$(dirname "$0")" && pwd)/$(basename "$0")}"

DATASET="${1:?dataset}"; TARGET="${2:?run_id or 'base'}"; shift 2

case "$DATASET" in
  kkp)     EXTRA=(--max_new_tokens 8192 --max_model_len 16384) ;;
  science) EXTRA=(--max_new_tokens 2048 --max_model_len 8192) ;;
  tooluse) EXTRA=(--max_new_tokens 1024) ;;
  *) echo "unknown dataset $DATASET"; exit 2 ;;
esac

if [ "$TARGET" = base ]; then
  MODEL="${MODEL:-Qwen/Qwen3-4B}"
  LABEL="${LABEL:-eval-$DATASET-base}"
  OUT="eval_results/$LABEL"
  python "eval_$DATASET.py" --model_path "$MODEL" --name "$LABEL" --output_dir "$OUT" \
    --seed 42 --temperature 0.0 "${EXTRA[@]}" "$@"
elif [ "$TARGET" = teacher ]; then
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then N="$1"; shift; else N=300; fi
  MODEL="${MODEL:-Qwen/Qwen3-4B}"
  LABEL="${LABEL:-eval-$DATASET-teacher-view}"
  OUT="eval_results/$LABEL"
  python "eval_$DATASET.py" --model_path "$MODEL" --name "$LABEL" --output_dir "$OUT" --teacher_view --num_samples "$N" \
    --seed 42 --temperature 0.0 "${EXTRA[@]}" "$@"
else
  RUN_DIR="experiments/runs/$TARGET"
  [ -f "$RUN_DIR/run.json" ] || { echo "no run record at $RUN_DIR"; exit 2; }
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then STEP="$1"; shift; CKPT="checkpoints/$TARGET/checkpoint-$STEP"
  else CKPT="$(ls -d checkpoints/"$TARGET"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"; STEP="${CKPT##*-}"; fi
  [ -d "$CKPT" ] || { echo "no checkpoint found for $TARGET"; exit 2; }
  MODEL="$(python -c "import json;print(json.load(open('$RUN_DIR/run.json'))['model']['name'])")"
  LABEL="${LABEL:-eval-$DATASET-$TARGET-s$STEP}"
  python "eval_$DATASET.py" --model_path "$MODEL" --adapter_path "$CKPT" --run_dir "$RUN_DIR" --name "$LABEL" \
    --output_dir "$CKPT/eval_$DATASET" --seed 42 --temperature 0.0 "${EXTRA[@]}" "$@"
fi
python -m explog table
