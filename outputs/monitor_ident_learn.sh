#!/bin/bash
# Server-side detached monitor for x0_exem_ident_learn.
# Writes progress to MONITOR.log every 30s; on completion records FINAL metrics + GATE.
RUN=outputs/train_runs/x0_exem_ident_learn
LOG=outputs/train_runs/x0_exem_ident_learn_MONITOR.log
echo "MONITOR START $(date)" > "$LOG"
for i in $(seq 1 240); do
  rows=$(wc -l < "$RUN/metrics.csv" 2>/dev/null || echo 0)
  alive=$(ps aux | grep -c '[l]ive2d_vla/train.py')
  last=$(tail -1 "$RUN/metrics.csv" 2>/dev/null)
  echo "t=$((i*30))s rows=$rows alive=$alive last: $last" >> "$LOG"
  if [ "$alive" -eq 0 ]; then echo "PROCESS ENDED" >> "$LOG"; break; fi
  if [ "$rows" -ge 31 ]; then echo "ALL EPOCHS DONE" >> "$LOG"; break; fi
  sleep 30
done
echo "=== FINAL metrics ===" >> "$LOG"
cat "$RUN/metrics.csv" >> "$LOG"
echo "=== GATE ===" >> "$LOG"
grep -i gate outputs/train_runs/x0_exem_ident_learn.log 2>/dev/null | tail -3 >> "$LOG"
echo "MONITOR DONE $(date)" >> "$LOG"
