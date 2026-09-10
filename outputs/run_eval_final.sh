#!/bin/bash
# Full-val offline rescoring of M / Q / R.
# One process per checkpoint: sharing a process shifts abs_mae by up to 7%.
# NOTE: no line continuations anywhere -- CRLF from the Windows side turns
# "cmd \<LF>" into "cmd \<CR><LF>" and the continuation silently breaks.
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs
GPU=${GPU:-cuda:2}

run_one() {
  local name="$1"; local run="$2"; local ckpt="$3"; local log="outputs/logs/eval_${name}_${ckpt}.log"
  echo "=== ${name} ${ckpt} $(date +%H:%M:%S) ==="
  .venv/bin/python outputs/eval_ckpt.py "${run}" --ckpt "${ckpt}" --device "${GPU}" --n_samples 0 > "${log}" 2>&1
  echo "exit=$? -> ${log}"
  tail -6 "${log}"
}

{
  run_one M abl_M_rangefix latest
  run_one M abl_M_rangefix best
  run_one Q abl_Q_std_v9 latest
  run_one Q abl_Q_std_v9 best
  run_one R abl_R_all_v9 latest
  run_one R abl_R_all_v9 best
  echo "ALL DONE $(date +%H:%M:%S)"
} > outputs/logs/eval_final_driver.log 2>&1

echo "eval chain finished; see outputs/logs/eval_final_driver.log"
