#!/bin/bash
# Quick progress check for the V11 arms.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
for r in abl_V11a_charloo abl_V11b_bank; do
  d=outputs/train_runs/$r
  [ -f "$d/metrics.csv" ] || continue
  echo "=== $r ==="
  echo "  last log: $(tail -1 outputs/logs/$r.log)"
  head -1 "$d/metrics.csv" | awk -F, '{printf "  %6s %10s %10s %10s %10s\n",$1,$4,$7,$3,$9}'
  tail -n +2 "$d/metrics.csv" | awk -F, 'NR%5==0 || NR==1{printf "  ep%-4s train=%.4f val=%.4f abs=%.4f exem=%.4f\n",$1,$2,$3,$4,$7}'
  tail -n +2 "$d/metrics.csv" | sort -t, -k4 -g | head -1 | awk -F, '{printf "  BEST so far: ep%s abs=%.4f (exem %.4f)\n",$1,$4,$7}'
done
