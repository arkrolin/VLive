#!/bin/bash
# Server-side detached monitor for x0_exem_ident run.
# Writes progress + final summary to x0_exem_ident_MONITOR.log regardless of SSH session lifetime.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
M=outputs/train_runs/x0_exem_ident
L=outputs/train_runs/x0_exem_ident_MONITOR.log
: > "$L"
echo "monitor start $(date)" >> "$L"
for i in $(seq 1 160); do
  sleep 30
  r=$(wc -l < "$M/metrics.csv" 2>/dev/null)
  a=$(ps aux | grep -c '[l]ive2d_vla/train.py')
  echo "t=$((i*30))s rows=$r alive=$a" >> "$L"
  tail -1 "$M/metrics.csv" 2>/dev/null >> "$L"
  if [ "$a" -eq 0 ]; then echo "PROCESS_ENDED" >> "$L"; break; fi
  if [ "$r" -ge 31 ]; then echo "ALL_DONE" >> "$L"; break; fi
done
echo "=== FINAL metrics ===" >> "$L"
cat "$M/metrics.csv" >> "$L"
echo "=== GATE ===" >> "$L"
grep -i gate "$M.log" | tail -3 >> "$L"
echo "monitor end $(date)" >> "$L"
