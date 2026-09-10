#!/bin/bash
# Full-val offline rescoring: M (baseline) vs Q (hygiene only) vs R (hygiene + expansion).
# One PROCESS PER CHECKPOINT - sharing a process shifts abs_mae by up to 7%.
# No backslash continuations anywhere (a stray trailing space after '\' silently
# ate the --n_samples flag last time).
# Whole chain is backgrounded inside the script so the ssh caller returns at once.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs

nohup bash -c '
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
run () {
  name="$1"; ck="$2"; dev="$3"
  echo "=== $name ckpt_$ck $(date +%H:%M:%S) ==="
  .venv/bin/python outputs/eval_ckpt.py "$name" --ckpt "$ck" --device "$dev" --n_samples 0 > "outputs/logs/eval_${name}_${ck}.log" 2>&1
  echo "$name/$ck exit=$?"
}
run abl_M_rangefix  best   cuda:1
run abl_M_rangefix  latest cuda:1
run abl_Q_std_v9    best   cuda:1
run abl_Q_std_v9    latest cuda:1
run abl_R_all_v9    best   cuda:2
run abl_R_all_v9    latest cuda:2
echo "DONE $(date +%H:%M:%S)"
' > outputs/logs/eval_all_driver.log 2>&1 </dev/null &
echo "launched eval chain (pid $!)"
