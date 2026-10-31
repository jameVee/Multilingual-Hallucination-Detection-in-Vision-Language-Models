#!/usr/bin/env bash
# Stage 1 pilot on nlp2 (rules: results/preregistration.md, R2). Qwen3-VL-8B, seed 42,
# ONE job at a time: train -> predict val -> score val_select. No test predictions here;
# those run only for conditions promoted to the full three-seed comparison.
# Safe to rerun after a crash: every step resumes or skips finished work.
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s pilot "bash thesis_v2/part2_synth/run_pilot.sh"
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
BB=qwen3vl8b
SEED=42
DATA=thesis_v2/part2_synth/data
LOGS=thesis_v2/runs/logs
mkdir -p "$LOGS"
QLOG="$LOGS/pilot.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
n_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
N_VAL=$(wc -l < data/val.jsonl)

run() {   # name, then the extra training files
  local NAME="$1"; shift
  local OUT="thesis_v2/runs/part2/${NAME}_seed${SEED}"
  log "== $NAME: train ($*)"
  python -m thesis_v2.part1_models.train --backbone $BB --seed $SEED --gpu_index "$GPU" --resume \
    --train_files data/train.jsonl "$@" --output_dir "$OUT" \
    2>&1 | tee -a "$LOGS/train_${NAME}_seed${SEED}.log" || true   # checked via train_state.json below
  if ! grep -q '"complete": true' "$OUT/train_state.json" 2>/dev/null; then
    log "$NAME: training did not complete -- stopping"; exit 1
  fi
  local F="$OUT/val_predictions.jsonl"
  if [ "$(n_lines "$F")" -lt "$N_VAL" ]; then
    log "== $NAME: predict val"
    python -m thesis_v2.part1_models.predict --backbone $BB --adapter "$OUT" --split val \
      --gpu_index "$GPU" --resume 2>&1 | tee -a "$LOGS/predict_${NAME}_seed${SEED}_val.log" || true
    if [ "$(n_lines "$F")" -lt "$N_VAL" ]; then
      log "$NAME: val predictions incomplete ($(n_lines "$F")/$N_VAL) -- stopping"; exit 1
    fi
  fi
  log "== $NAME: score val_select"
  python -m thesis_v2.common.evaluate --pred "$F" --split val_select --name "p2_${NAME}_seed${SEED}" 2>&1 | tee -a "$QLOG"
}

# most informative first; the baseline is p1_qwen3vl8b_seed42 (val_select 0.4269)
run C4x "$DATA/C4_candidates.jsonl" "$DATA/C4x_claims.jsonl"
run C4 "$DATA/C4_candidates.jsonl"
run C1 "$DATA/C1_upsample.jsonl"
log "pilot finished"
