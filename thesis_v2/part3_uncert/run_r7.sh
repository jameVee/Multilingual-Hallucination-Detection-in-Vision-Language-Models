#!/usr/bin/env bash
# R7 queue on nlp2 (results/preregistration.md): second frozen judge for the probe.
#   1. Gemma-3-12B features for val, test, train (chunked; resumes inside a split)
#   2. compaction: PCA on train, compact files with full-coverage check
#   3. multi-label CRF probe, seeds 42/123/2024
# ONE GPU job at a time. Safe to rerun after a crash: finished steps leave a marker.
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s r7 "bash thesis_v2/part3_uncert/run_r7.sh"
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
BB=gemma3_12b
LOGS=thesis_v2/runs/logs
DONE=thesis_v2/runs/r7/done
mkdir -p "$LOGS" "$DONE"
QLOG="$LOGS/r7.log"
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

for SPLIT in val test train; do
  step "extract_$SPLIT" "r7_extract_$SPLIT" \
    python -m thesis_v2.part3_uncert.extract_features_judge2 --backbone "$BB" --split "$SPLIT" --gpu_index "$GPU"
done
step compact r7_compact python -m thesis_v2.part3_uncert.features --build --source "$BB"
for SEED in 42 123 2024; do
  step "probe_$SEED" "r7_probe_$SEED" \
    python -m thesis_v2.part3_uncert.crf_probe --seed "$SEED" --gpu_index "$GPU" --multilabel --source "$BB"
done
log "R7 queue finished"
