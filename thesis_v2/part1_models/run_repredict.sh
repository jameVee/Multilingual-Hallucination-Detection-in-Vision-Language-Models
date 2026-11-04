#!/usr/bin/env bash
# Re-predict val with raw generations saved and the dual-label decoder (common/span_io).
# ONE GPU job at a time. For each run: the old val_predictions.jsonl (no "raw") is kept as
# val_predictions_olddecoder.jsonl; the new greedy run decoded the OLD way must reproduce it
# (checked, logged); then both val halves are scored as dec2_<name>.
# Safe to rerun after a crash: predictions resume by id.
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s repredict "bash thesis_v2/part1_models/run_repredict.sh"
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
LOGS=thesis_v2/runs/logs
mkdir -p "$LOGS"
QLOG="$LOGS/repredict.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
N_VAL=$(wc -l < data/val.jsonl)

# the 3a test samples stopped at 252 records were decoded the old way: set aside, never resumed
S=thesis_v2/runs/part1/qwen3vl8b_seed42/test_samples8_T0.7.jsonl
[ -f "$S" ] && mv "$S" "${S%.jsonl}_olddecoder_partial.jsonl"

# name : run dir (C0 first -- it shows how often the model emits dual labels at all)
for ITEM in p1_qwen3vl8b_seed42:part1/qwen3vl8b_seed42 p2_C4x_seed42:part2/C4x_seed42 \
            p2_C1_seed42:part2/C1_seed42 p2_C4_seed42:part2/C4_seed42; do
  NAME="${ITEM%%:*}"; D="thesis_v2/runs/${ITEM#*:}"; F="$D/val_predictions.jsonl"; OLD="$D/val_predictions_olddecoder.jsonl"
  if [ -f "$F" ] && ! head -1 "$F" | grep -q '"raw"'; then mv "$F" "$OLD"; fi
  log "== $NAME: predict val (raw saved)"
  python -m thesis_v2.part1_models.predict --backbone qwen3vl8b --adapter "$D" --split val \
    --gpu_index "$GPU" --resume 2>&1 | tee -a "$LOGS/repredict_${NAME}_val.log" || true
  if [ "$(wc -l < "$F")" -lt "$N_VAL" ]; then
    log "$NAME: val predictions incomplete ($(wc -l < "$F")/$N_VAL) -- stopping"; exit 1
  fi
  python -m thesis_v2.common.redecode --pred "$F" --decoder old --compare "$OLD" 2>&1 | tee -a "$QLOG"
  for SPLIT in val_select val_confirm; do
    python -m thesis_v2.common.evaluate --pred "$F" --split $SPLIT --name "dec2_${NAME}" 2>&1 | tee -a "$QLOG"
  done
done
log "repredict finished"
