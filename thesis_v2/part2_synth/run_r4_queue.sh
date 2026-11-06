#!/usr/bin/env bash
# R4 queue on nlp2 (results/preregistration.md, R4), ONE GPU job at a time, started after
# the R3 queue (tmux session "fix") has finished. Order = what each result decides:
#   1. C4x-b seed 42: train (3,020-step budget), predict val, score val_select
#   2. C0 seeds 123, 2024: train, greedy val + test, score both val halves
#   3. C0 seeds 123, 2024: 8 sampled decodes (T=0.7) on val and test
#   4. C1-b seed 42: train (3,020-step budget), predict val, score val_select
# Safe to rerun after a crash: training and prediction resume, finished steps leave a
# marker in runs/r4/done/.
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s r4 "bash thesis_v2/part2_synth/run_r4_queue.sh"
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
BB=qwen3vl8b
BUDGET=3020
DATA=thesis_v2/part2_synth/data
LOGS=thesis_v2/runs/logs
DONE=thesis_v2/runs/r4/done
mkdir -p "$LOGS" "$DONE"
QLOG="$LOGS/r4.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
n_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
N_VAL=$(wc -l < data/val.jsonl)
N_TEST=$(wc -l < data/test.jsonl)

train() {   # output dir, seed, then extra train.py args
  local OUT="$1" SEED="$2"; shift 2
  local TAG; TAG=$(basename "$OUT")
  if [ -f "$DONE/train_$TAG" ]; then log "$TAG: trained earlier -- skipping"; return; fi
  log "== $TAG: train"
  python -u -m thesis_v2.part1_models.train --backbone $BB --seed "$SEED" --gpu_index "$GPU" --resume \
    --output_dir "$OUT" "$@" 2>&1 | tee -a "$LOGS/train_${TAG}.log" || true
  if ! grep -q '"complete": true' "$OUT/train_state.json" 2>/dev/null; then
    log "$TAG: training did not complete -- stopping"; exit 1
  fi
  local STEPS; STEPS=$(python -c "import json;print(json.load(open('$OUT/train_stats.json'))['optimizer_steps'])")
  if [ "$STEPS" != "$BUDGET" ]; then log "$TAG: $STEPS optimizer steps, expected $BUDGET -- stopping"; exit 1; fi
  touch "$DONE/train_$TAG"
}

predict() {   # output dir, split, then extra predict.py args; F = expected output file
  local OUT="$1" SPLIT="$2" F="$3"; shift 3
  local WANT; WANT=$([ "$SPLIT" = val ] && echo "$N_VAL" || echo "$N_TEST")
  if [ "$(n_lines "$F")" -ge "$WANT" ]; then log "$(basename "$OUT"): $(basename "$F") complete -- skipping"; return; fi
  log "== $(basename "$OUT"): $(basename "$F")"
  python -m thesis_v2.part1_models.predict --backbone $BB --adapter "$OUT" --split "$SPLIT" \
    --gpu_index "$GPU" --resume "$@" 2>&1 | tee -a "$LOGS/predict_$(basename "$OUT")_$(basename "$F" .jsonl).log" || true
  if [ "$(n_lines "$F")" -lt "$WANT" ]; then
    log "$(basename "$OUT"): $(basename "$F") incomplete ($(n_lines "$F")/$WANT) -- stopping"; exit 1
  fi
}

score() {   # prediction file, score name, splits...
  local F="$1" NAME="$2"; shift 2
  for S in "$@"; do
    python -m thesis_v2.common.evaluate --pred "$F" --split "$S" --name "$NAME" 2>&1 | tee -a "$QLOG"
  done
}

synth() {   # condition name, extra train files...
  local NAME="$1"; shift
  local OUT="thesis_v2/runs/part2/${NAME}_seed42"
  train "$OUT" 42 --step_budget $BUDGET --train_files data/train.jsonl "$@"
  predict "$OUT" val "$OUT/val_predictions.jsonl"
  score "$OUT/val_predictions.jsonl" "p2_${NAME}_seed42" val_select
}

while tmux has-session -t fix 2>/dev/null; do sleep 300; done
log "fix session finished -- starting"

synth C4xb "$DATA/C4_candidates.jsonl" "$DATA/C4x_claims.jsonl" "$DATA/R4_C4xb_clean.jsonl"

for SEED in 123 2024; do
  OUT="thesis_v2/runs/part1/${BB}_seed${SEED}"
  if [ ! -f "$DONE/train_$(basename "$OUT")" ]; then
    log "== $(basename "$OUT"): train (C0 recipe, no budget)"
    python -u -m thesis_v2.part1_models.train --backbone $BB --seed "$SEED" --gpu_index "$GPU" --resume \
      --output_dir "$OUT" 2>&1 | tee -a "$LOGS/train_$(basename "$OUT").log" || true
    grep -q '"complete": true' "$OUT/train_state.json" 2>/dev/null || { log "$OUT: training did not complete -- stopping"; exit 1; }
    touch "$DONE/train_$(basename "$OUT")"
  fi
  predict "$OUT" val "$OUT/val_predictions.jsonl"
  score "$OUT/val_predictions.jsonl" "dec2_p1_${BB}_seed${SEED}" val_select val_confirm
  predict "$OUT" test "$OUT/test_predictions.jsonl"
done

for SEED in 123 2024; do
  OUT="thesis_v2/runs/part1/${BB}_seed${SEED}"
  predict "$OUT" val "$OUT/val_samples8_T0.7.jsonl" --num_samples 8 --temperature 0.7
  predict "$OUT" test "$OUT/test_samples8_T0.7.jsonl" --num_samples 8 --temperature 0.7
done

synth C1b "$DATA/C1_upsample.jsonl" "$DATA/R4_C1b_clean.jsonl"
log "r4 queue finished"
