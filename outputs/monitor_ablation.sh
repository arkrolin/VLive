#!/usr/bin/env bash
# Server-side detached monitor for the P1 ablation (3 arms).
# Writes progress to a log file on disk so it survives SSH drops.
# Exits when all three runs are no longer alive.
set -u
cd /root/work/nlp/xjzhao13/lijie_llama/VLive || exit 1

RUNS="abl_A_regress abl_B_ddpm_t1000 abl_C_ddpm_t200"
LOG=outputs/train_runs/ABLATION_MONITOR.log
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

log "monitor started; runs=$RUNS"

for i in $(seq 1 200); do
  alive=0
  for r in $RUNS; do
    if ps aux | grep -q "[o]ut_dir outputs/train_runs/$r"; then
      alive=$((alive + 1))
    fi
  done
  log "--- tick $i : arms alive=$alive ---"
  for r in $RUNS; do
    f="outputs/train_runs/$r/metrics.csv"
    if [ -f "$f" ]; then
      rows=$(wc -l < "$f")
      last=$(tail -1 "$f")
      log "$r rows=$((rows - 1)) last: $last"
    else
      log "$r (no metrics yet)"
    fi
  done
  if [ "$alive" -eq 0 ]; then
    log "ALL ARMS FINISHED"
    break
  fi
  sleep 60
done

log "=== FINAL METRICS ==="
for r in $RUNS; do
  f="outputs/train_runs/$r/metrics.csv"
  log "----- $r -----"
  if [ -f "$f" ]; then
    while IFS= read -r line; do log "$line"; done < "$f"
  fi
done
log "=== GATES ==="
for r in $RUNS; do
  lf="outputs/train_runs/abl_${r%%_*}.log"
  if [ -f "$lf" ]; then
    log "----- $r -----"
    grep -E "TRAINING GATE|train_loss|val_loss|overfit|val_rel_mae_f|exem prior|BEATS EXEM|val_abs_mae|GATE:" "$lf" | tail -n 10 >> "$LOG" 2>/dev/null
  fi
done
log "MONITOR DONE"
