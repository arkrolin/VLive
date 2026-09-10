#!/bin/bash
# Full-val offline rescoring of the two V9 arms while they are still training.
# One process per checkpoint (sharing a process shifts abs_mae by up to 7%).
# cuda:2 has ~52 GB free (Q occupies 28 GB of the 80 GB A800).
#
# The whole chain is backgrounded INSIDE the script so the caller returns
# immediately (backgrounding over ssh keeps the session open and the client
# gets SIGTERMed).
cd /root/work/nlp/xjzhao13/lijie_llama/VLive
mkdir -p outputs/logs

nohup bash -c '
  cd /root/work/nlp/xjzhao13/lijie_llama/VLive
  echo "=== Q $(date +%H:%M:%S) ===" 
  .venv/bin/python outputs/eval_ckpt.py abl_Q_std_v9 --ckpt latest \
    --device cuda:2 --n_samples 0 > outputs/logs/eval_Q_latest.log 2>&1
  echo "Q exit=$?"
  echo "=== R $(date +%H:%M:%S) ==="
  .venv/bin/python outputs/eval_ckpt.py abl_R_all_v9 --ckpt latest \
    --device cuda:2 --n_samples 0 > outputs/logs/eval_R_latest.log 2>&1
  echo "R exit=$?"
  echo "DONE $(date +%H:%M:%S)"
' > outputs/logs/eval_QR_driver.log 2>&1 </dev/null &
echo "launched eval chain (pid $!)"
