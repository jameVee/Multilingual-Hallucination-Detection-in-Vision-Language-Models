#!/usr/bin/env bash
# Part 1 queue on nlp2: for each backbone, ONE job at a time (two concurrent 3090 jobs
# crashed the machine on 2026-10-26):
#   train (seed 42) -> predict val -> predict test -> score val_select
#
# Safe to rerun after a crash/reset -- every step resumes or skips finished work:
#   train   --resume continues from the last checkpoint ("already complete" if done)
#   predict --resume skips ids already written; skipped entirely when the file is complete
# It first waits for any training/prediction job already running (no overlap).
#
# Usage (on nlp2, from the repo root):
#   tmux new-session -d -s queue "bash thesis_v2/part1_models/run_queue.sh"
#   tail -f thesis_v2/runs/logs/queue.log
# Optional: GPU=1 BACKBONES="internvl3_8b pixtral12b" bash thesis_v2/part1_models/run_queue.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
GPU="${GPU:-0}"
BACKBONES="${BACKBONES:-qwen3vl8b internvl3_8b gemma3_12b pixtral12b}"
SEED=42
LOGS=thesis_v2/runs/logs
mkdir -p "$LOGS"
QLOG="$LOGS/queue.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

# anchored on the python process itself: the tmux server's command line also contains
# "python -m thesis_v2..." for as long as any session lives, so an unanchored match
# would wait forever
JOB_RE='^(\S*/)?python[0-9.]* -m thesis_v2\.part1_models\.(train|predict)'
while pgrep -f "$JOB_RE" > /dev/null; do
  log "waiting: another Part 1 job is running ($(pgrep -fa "$JOB_RE" | head -1 | cut -c1-120))"
  sleep 300
done

n_lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
N_VAL=$(wc -l < data/val.jsonl)
N_TEST=$(wc -l < data/test.jsonl)

for B in $BACKBONES; do
  OUT="thesis_v2/runs/part1/${B}_seed${SEED}"
  log "== $B: train"
  python -m thesis_v2.part1_models.train --backbone "$B" --seed $SEED --gpu_index "$GPU" --resume \
    2>&1 | tee -a "$LOGS/train_${B}_seed${SEED}.log" || true   # checked via train_state.json below
  if ! grep -q '"complete": true' "$OUT/train_state.json" 2>/dev/null; then
    log "$B: training did not complete -- stopping the queue"; exit 1
  fi

  for SPLIT in val test; do
    F="$OUT/${SPLIT}_predictions.jsonl"
    WANT=$([ "$SPLIT" = val ] && echo "$N_VAL" || echo "$N_TEST")
    if [ "$(n_lines "$F")" -ge "$WANT" ]; then
      log "$B: $SPLIT predictions complete ($WANT) -- skipping"; continue
    fi
    log "== $B: predict $SPLIT"
    python -m thesis_v2.part1_models.predict --backbone "$B" --adapter "$OUT" --split "$SPLIT" \
      --gpu_index "$GPU" --resume 2>&1 | tee -a "$LOGS/predict_${B}_seed${SEED}_${SPLIT}.log" || true
    if [ "$(n_lines "$F")" -lt "$WANT" ]; then
      log "$B: $SPLIT predictions incomplete ($(n_lines "$F")/$WANT) -- stopping the queue"; exit 1
    fi
  done

  # val_select only: val_confirm is looked at when a decision is confirmed, not before
  log "== $B: score val_select"
  python -m thesis_v2.common.evaluate --pred "$OUT/val_predictions.jsonl" --split val_select \
    --name "p1_${B}_seed${SEED}" 2>&1 | tee -a "$QLOG"
  if [ "$B" = qwen3vl8b ]; then
    log "pipeline check: compare with solution 02 reference val_select Cor+Lbl 0.4214 (must be within ~0.01)"
  fi
done
log "queue finished: $BACKBONES"
