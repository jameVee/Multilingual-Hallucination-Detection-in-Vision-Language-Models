#!/usr/bin/env bash
# R3 queue on nlp2 (results/preregistration.md), ONE GPU job at a time, started after the
# val re-prediction queue (tmux session "repredict") has finished:
#   1. 3b v2: CRF selftest, multi-label probe seeds 42/123/2024
#   2. 3a v2: 8 val decodes of the C0 adapter, raw saved, new decoder
#   3. C0 test predictions with the new decoder (new baseline's test file)
#   4. 3a v2: 8 test decodes
# Files written by the old decoder (no "raw") are kept as *_olddecoder.jsonl, never resumed.
# Safe to rerun after a crash: finished steps leave a marker in runs/fix/done/.
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s fix "bash thesis_v2/part3_uncert/run_fix_queue.sh"
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
LOGS=thesis_v2/runs/logs
DONE=thesis_v2/runs/fix/done
C0=thesis_v2/runs/part1/qwen3vl8b_seed42
mkdir -p "$LOGS" "$DONE"
QLOG="$LOGS/fix.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

step() {   # marker name, log name, command...
  local NAME="$1" LOGF="$2"; shift 2
  if [ -f "$DONE/$NAME" ]; then log "$NAME: done earlier -- skipping"; return; fi
  log "== $NAME"
  if "$@" 2>&1 | tee -a "$LOGS/$LOGF.log"; then
    touch "$DONE/$NAME"
  else
    log "$NAME failed -- stopping"; exit 1
  fi
}
set_aside_old() {   # a prediction file without raw generations -> *_olddecoder.jsonl
  if [ -f "$1" ] && ! head -1 "$1" | grep -q '"raw"'; then mv "$1" "${1%.jsonl}_olddecoder.jsonl"; fi
}

while tmux has-session -t repredict 2>/dev/null; do sleep 300; done
log "repredict session finished -- starting"

step crf_selftest fix_crf python -m thesis_v2.part3_uncert.crf_probe --selftest
for SEED in 42 123 2024; do
  step "crfml_seed$SEED" "fix_crfml_seed$SEED" python -m thesis_v2.part3_uncert.crf_probe --multilabel \
    --seed $SEED --gpu_index "$GPU"
done
set_aside_old "$C0/val_samples8_T0.7.jsonl"
step sample_val fix_sample_val python -m thesis_v2.part1_models.predict --backbone qwen3vl8b --adapter "$C0" \
  --split val --num_samples 8 --temperature 0.7 --gpu_index "$GPU" --resume
set_aside_old "$C0/test_predictions.jsonl"
step c0_test fix_c0_test python -m thesis_v2.part1_models.predict --backbone qwen3vl8b --adapter "$C0" \
  --split test --gpu_index "$GPU" --resume
step sample_test fix_sample_test python -m thesis_v2.part1_models.predict --backbone qwen3vl8b --adapter "$C0" \
  --split test --num_samples 8 --temperature 0.7 --gpu_index "$GPU" --resume
log "fix queue finished"
