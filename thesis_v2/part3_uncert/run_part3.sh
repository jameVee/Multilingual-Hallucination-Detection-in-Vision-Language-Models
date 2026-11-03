#!/usr/bin/env bash
# Part 3 queue on nlp2, ONE GPU job at a time. Safe to rerun after a crash: finished steps
# leave a marker in runs/part3/done/ and are skipped; predictions resume by id.
#   1. 3.0  clean-record features with the Qwen judge (val, then train)
#   2. 3b   compact features (PCA, full-coverage check), CRF selftest, CRF seeds 42/123/2024
#   3. 3a   8 sampled decodes per val record (Part 1 winner, seed 42), then STOP:
#           the diversity check runs on these before any test sampling (pre-registered)
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s part3 "bash thesis_v2/part3_uncert/run_part3.sh"
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
LOGS=thesis_v2/runs/logs
DONE=thesis_v2/runs/part3/done
mkdir -p "$LOGS" "$DONE"
QLOG="$LOGS/part3.log"
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

step extract_val   part3_extract_val   python -m thesis_v2.part3_uncert.extract_features_all --split val --gpu_index "$GPU"
step extract_train part3_extract_train python -m thesis_v2.part3_uncert.extract_features_all --split train --gpu_index "$GPU"
step features      part3_features      python -m thesis_v2.part3_uncert.features --build
step crf_selftest  part3_crf           python -m thesis_v2.part3_uncert.crf_probe --selftest
for SEED in 42 123 2024; do
  step "crf_seed$SEED" "part3_crf_seed$SEED" python -m thesis_v2.part3_uncert.crf_probe --seed $SEED --gpu_index "$GPU"
done
step sample_val part3_sample_val python -m thesis_v2.part1_models.predict --backbone qwen3vl8b \
  --adapter thesis_v2/runs/part1/qwen3vl8b_seed42 --split val --num_samples 8 --temperature 0.7 \
  --gpu_index "$GPU" --resume
log "part3 queue finished (test sampling waits for the 3a diversity check)"
